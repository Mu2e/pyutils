#! /usr/bin/env python
import uproot
import os
import subprocess
from . import _env_manager
from .pylogger import Logger

class Reader:
    """Unified interface for reading files, either locally or remotely"""
    
    def __init__(self, use_remote=False, location="tape", schema="root", verbosity=1):
        """Initialise the reader
        
        Args:
            use_remote (bool, opt): Whether to use remote access methods
            location (str, opt): File location for remote files: 'tape' (default), 'disk', 'scratch', 'nersc' 
            schema (str, opt): Schema for remote file path: 'root' (default), 'http', 'path', 'dcap', 'sam'
            verbosity (int, opt): Level of output detail (0: errors only, 1: info & warnings, 2: max)
        """
        self.use_remote = use_remote # access files on /pnfs from EAF
        self.location = location
        self.schema = schema

        # Start logger 
        self.logger = Logger( 
            print_prefix = "[pyread]", 
            verbosity = verbosity
        )

        # Setup and validation for remote reading
        if self.use_remote:
            #  Ensure mdh environment
            _env_manager.ensure_environment()  
            # Check arguments
            self.valid_locations = ["tape", "disk", "scratch", "nersc"]
            if self.location not in self.valid_locations:
                self.logger.log(f"Location '{location}' may not be valid. Expected one of {self.valid_locations}", "warning")
            self.valid_schemas = ["root", "http", "path", "dcap", "sam"]
            if self.schema not in self.valid_schemas:
                self.logger.log(f"Schema '{schema}' may not be valid. Expected one of {self.valid_schemas}", "warning")

    def read_file(self, file_path):
        """Read a file using the appropriate method
        
        Args:
            file_path: Path to the file
            
        Returns:
            uproot file object
        """
        if self.use_remote:
            return self._read_remote_file(file_path)
        else:
            return self._read_file(file_path)
    
    def _read_file(self, file_path):
        """Open file with uproot"""
        try: 
            file = uproot.open(file_path)
            self.logger.log(f"Opened {file_path}", "success")
            return file
        except Exception as e:
            self.logger.log(f"Exception while opening {file_path}: {e}", "warning")
            raise # propagate exception up

    def _read_remote_file(self, file_path):
        """Open a file from /pnfs via mdh - NO FALLBACKS"""
        self.logger.log(f"Opening remote file: {file_path}", "info")
        # Try the specified location 
        return self._attempt_remote_read(file_path, self.location)
    
    def _attempt_remote_read(self, file_path, location):
        """Attempt to read remote file with specific location"""
        commands = f"mdh print-url {file_path} -l {location} -s {self.schema}"
        
        this_file_path = subprocess.check_output(
            commands,
            shell=True,
            universal_newlines=True, 
            stderr=subprocess.DEVNULL,
            timeout=30
        ).strip()
        
        self.logger.log(f"Created file path: {this_file_path}", "info")
        
        # Read the file
        return self._read_file(this_file_path)

        # Previously I had a fallback method which tried to read from multiple locations,
        # but I think it is far easier to debug if you simply let the process fail than
        # to deal with fallout from overwhelming the network with subprocess calls 