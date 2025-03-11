import os
import json
import torch
import hashlib
import pickle
from typing import Dict, Any, Optional, Type, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_store')

class ModelStore:
    """
    A class for managing ML model storage, versioning, and loading with architecture validation.
    
    This class helps prevent model architecture mismatches by storing model architecture
    information alongside weights and validating compatibility during loading.
    """
    
    def __init__(self, base_dir: str = "~/models"):
        """
        Initialize the model store.
        
        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Track registered model classes
        self._registered_models = {}
        
    def register_model_class(self, model_type: str, model_class: Type):
        """
        Register a model class with the store.
        
        Args:
            model_type: String identifier for the model type
            model_class: The model class (not instance)
        """
        self._registered_models[model_type] = model_class
        logger.info(f"Registered model class: {model_type}")
    
    def _get_model_path(self, model_name: str, version: str = "latest") -> str:
        """
        Get the full path for a model.
        
        Args:
            model_name: Name of the model
            version: Version of the model (default: "latest")
            
        Returns:
            Full path to the model directory
        """
        # First check if the model file exists directly in base_dir
        direct_path = os.path.join(self.base_dir, model_name)
        if os.path.exists(direct_path) or os.path.exists(direct_path + ".pt"):
            return direct_path
            
        # If not found directly, look for versioned directory structure
        if version == "latest":
            model_dir = os.path.join(self.base_dir, model_name)
                
            versions = [d for d in os.listdir(model_dir) 
                      if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('v')]
            
            if not versions:
                # If no versioned directories found, but model_dir exists, return it
                if os.path.exists(model_dir):
                    return model_dir
                raise FileNotFoundError(f"No versions found for model '{model_name}'")
                
            # Sort versions and get the latest
            versions.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0, reverse=True)
            version = versions[0]
        
        versioned_path = os.path.join(self.base_dir, model_name, version)
        if os.path.exists(versioned_path):
            return versioned_path
            
        raise FileNotFoundError(f"Model '{model_name}' not found in any location")
    
    def _compute_architecture_hash(self, model) -> str:
        """
        Compute a hash of the model architecture.
        
        Args:
            model: The model instance
            
        Returns:
            Hash string representing the architecture
        """
        # Get model architecture as string representation
        arch_str = str(model)
        
        # Create a hash of the architecture
        return hashlib.md5(arch_str.encode()).hexdigest()
    
    def _extract_model_metadata(self, model, **kwargs) -> Dict[str, Any]:
        """
        Extract metadata from a model.
        
        Args:
            model: The model instance
            **kwargs: Additional metadata to store
            
        Returns:
            Dictionary of model metadata
        """
        metadata = {
            'architecture_hash': self._compute_architecture_hash(model),
            'model_type': model.__class__.__name__,
            'architecture_summary': str(model),
        }
        
        # Add any additional metadata
        metadata.update(kwargs)
        
        return metadata
    
    def save(self, model, model_name: str, optimizer=None, 
             version: Optional[str] = None, tag: Optional[str] = None, **kwargs) -> str:
        """
        Save a model with its architecture information, metadata, and optional tag.
        
        Args:
            model: The model to save
            model_name: Name to save the model under
            optimizer: Optional optimizer to save
            version: Version string (if None, auto-increment)
            tag: Optional tag for the model (e.g., 'best', 'stable')
            **kwargs: Additional metadata to store
            
        Returns:
            Path where the model was saved
        """
        # Create models root directory if it doesn't exist
        models_root = os.path.join(self.base_dir, 'models')
        os.makedirs(models_root, exist_ok=True)
        
        # Create model-specific directory
        model_dir = os.path.join(models_root, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Determine version
        if version is None:
            # Auto-increment version
            existing_versions = [d for d in os.listdir(model_dir) 
                               if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('v')]
            
            if not existing_versions:
                version = "v1"
            else:
                # Extract version numbers and find the highest
                version_nums = [int(v[1:]) for v in existing_versions if v[1:].isdigit()]
                if not version_nums:
                    version = "v1"
                else:
                    version = f"v{max(version_nums) + 1}"
        
        # Create version directory
        version_dir = os.path.join(model_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Extract metadata
        metadata = self._extract_model_metadata(model, **kwargs)
        
        # Save model state dict
        model_path = os.path.join(version_dir, "model.pt")
        save_dict = {
            'model_state_dict': model.state_dict(),
        }
        
        # Add optimizer if provided
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, model_path)
        
        # Save metadata
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save hyperparameters if provided
        if 'hyperparameters' in kwargs:
            hyperparams_path = os.path.join(version_dir, "hyperparameters.json")
            with open(hyperparams_path, 'w') as f:
                json.dump(kwargs['hyperparameters'], f, indent=2)
        
        logger.info(f"Saved model {model_name} version {version} to {version_dir}")
        return version_dir
    
    def load(self, model_instance, model_name: str, version: str = "latest", 
             optimizer=None, strict: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model with architecture validation.
        
        Args:
            model_instance: An instance of the model to load weights into
            model_name: Name of the model to load
            version: Version to load (default: "latest")
            optimizer: Optional optimizer to load state into
            strict: Whether to strictly enforce architecture matching
            
        Returns:
            Tuple of (loaded_model, metadata)
        """
        # Get model directory        
        try:
            model_dir = self._get_model_path(model_name, version)
        except FileNotFoundError as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            logger.warning(f"No metadata found for model {model_name} version {version}")
            metadata = {}
        else:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Validate architecture if metadata exists
        if 'architecture_hash' in metadata:
            current_hash = self._compute_architecture_hash(model_instance)
            stored_hash = metadata['architecture_hash']
            
            if current_hash != stored_hash:
                error_msg = (
                    f"Model architecture mismatch!\n"
                    f"Current architecture hash: {current_hash}\n"
                    f"Stored architecture hash: {stored_hash}\n"
                    f"Current architecture:\n{str(model_instance)}\n"
                    f"Stored architecture summary:\n{metadata.get('architecture_summary', 'Not available')}\n"
                )
                
                if strict:
                    logger.error(error_msg)
                    raise ValueError("Model architecture mismatch. Use strict=False to override.")
                else:
                    logger.warning(error_msg + "\nContinuing with load despite mismatch (strict=False)")
        
        # Load model weights
        model_path = os.path.join(model_dir, "v1/model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Load model state dict
        try:
            model_instance.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Successfully loaded model {model_name} version {version}")
        except Exception as e:
            logger.error(f"Failed to load model state dict: {str(e)}")
            raise ValueError(f"Error loading model state dict: {str(e)}")
        
        # Load optimizer if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Successfully loaded optimizer state")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {str(e)}")
        
        # Load hyperparameters if available
        hyperparams_path = os.path.join(model_dir, "hyperparameters.json")
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, 'r') as f:
                metadata['hyperparameters'] = json.load(f)
        
        return model_instance, metadata
    
    def load_by_type(self, model_type: str, model_name: str, version: str = "latest", 
                    **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model by its registered type.
        
        Args:
            model_type: Type of model to load (must be registered)
            model_name: Name of the model to load
            version: Version to load (default: "latest")
            **kwargs: Additional arguments to pass to model constructor
            
        Returns:
            Tuple of (loaded_model, metadata)
        """
        if model_type not in self._registered_models:
            raise ValueError(f"Model type '{model_type}' not registered. "
                           f"Available types: {list(self._registered_models.keys())}")
        
        # Create model instance
        model_class = self._registered_models[model_type]
        model_instance = model_class(**kwargs)
        
        # Load the model
        return self.load(model_instance, model_name, version)
    
    def list_models(self) -> Dict[str, list]:
        """
        List all available models and their versions.
        
        Returns:
            Dictionary mapping model names to lists of available versions
        """
        result = {}
        
        if not os.path.exists(self.base_dir):
            return result
        
        for model_name in os.listdir(self.base_dir):
            model_dir = os.path.join(self.base_dir, model_name)
            if os.path.isdir(model_dir):
                versions = [d for d in os.listdir(model_dir) 
                          if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('v')]
                versions.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
                result[model_name] = versions
        
        return result
    
    def get_metadata(self, model_name: str, version: str = "latest") -> Dict[str, Any]:
        """
        Get metadata for a model without loading it.
        
        Args:
            model_name: Name of the model
            version: Version of the model (default: "latest")
            
        Returns:
            Dictionary of model metadata
        """
        model_dir = self._get_model_path(model_name, version)
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            logger.warning(f"No metadata found for model {model_name} version {version}")
            return {}
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """
        Delete a model or specific version.
        
        Args:
            model_name: Name of the model to delete
            version: Specific version to delete (if None, delete all versions)
            
        Returns:
            True if deletion was successful
        """
        import shutil
        
        model_dir = os.path.join(self.base_dir, model_name)
        if not os.path.exists(model_dir):
            logger.warning(f"Model {model_name} not found")
            return False
        
        if version is None:
            # Delete entire model directory
            shutil.rmtree(model_dir)
            logger.info(f"Deleted model {model_name} (all versions)")
            return True
        else:
            # Delete specific version
            version_dir = os.path.join(model_dir, version)
            if not os.path.exists(version_dir):
                logger.warning(f"Version {version} of model {model_name} not found")
                return False
            
            shutil.rmtree(version_dir)
            logger.info(f"Deleted model {model_name} version {version}")
            return True