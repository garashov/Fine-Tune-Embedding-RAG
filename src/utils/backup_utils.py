import logging
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.config.config import (
    HF_PUSH_BACKUP_CREATE_BACKUP,
    HF_PUSH_BACKUP_BACKUP_DIR,
    HF_PUSH_BACKUP_BACKUP_NAME,
    HF_PUSH_BACKUP_MAX_BACKUPS,
)


class BackupManager:
    """
    Manages model backups before pushing
    Can be moved to utils/backup_utils.py
    """
    
    def __init__(
        self,
        create_backup: bool = HF_PUSH_BACKUP_CREATE_BACKUP,
        backup_dir: str = HF_PUSH_BACKUP_BACKUP_DIR,
        backup_name: str = HF_PUSH_BACKUP_BACKUP_NAME,
        max_backups: int = HF_PUSH_BACKUP_MAX_BACKUPS,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize backup manager"""
        self.create_backup = create_backup
        self.backup_dir = Path(backup_dir)
        self.backup_name = backup_name
        self.max_backups = max_backups
        self.logger = logger or logging.getLogger(__name__)
    
    def backup_model(self, model_path: Path) -> Optional[Path]:
        """
        Create backup of model
        
        Args:
            model_path: Path to model to backup
        
        Returns:
            Path to backup or None if backup disabled
        """
        if not self.create_backup:
            self.logger.info("Backup disabled, skipping")
            return None
        
        self.logger.info("\n" + "-" * 80)
        self.logger.info("Creating backup...")
        self.logger.info("-" * 80)
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate backup name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_path.name
        backup_name = self.backup_name.format(
            model_name=model_name,
            timestamp=timestamp
        )
        backup_path = self.backup_dir / backup_name
        
        try:
            # Copy model directory
            shutil.copytree(model_path, backup_path)
            self.logger.info(f"✓ Backup created: {backup_path}")
            
            # Clean old backups
            self._cleanup_old_backups()
            
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {str(e)}")
            raise
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backups beyond max_backups limit"""
        if self.max_backups <= 0:
            return
        
        # Get all backups sorted by modification time
        backups = sorted(
            self.backup_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove old backups
        for backup in backups[self.max_backups:]:
            try:
                if backup.is_dir():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()
                self.logger.info(f"Removed old backup: {backup.name}")
            except Exception as e:
                self.logger.warning(f"Failed to remove backup {backup.name}: {str(e)}")

