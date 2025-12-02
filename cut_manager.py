# from cut_manager import CutManager

import json
import awkward as ak
import csv
from pyutils.pylogger import Logger

class CutManager:
    """Class to manage analysis cuts"""
    
    def __init__(self, verbosity=1):
        """Initialise 
        
        Args:
            verbosity (int, optional): Printout level (0: minimal, 1: normal, 2: detailed)
        """
        # Init cut object
        self.cuts = {}
        # Start logger
        self.logger = Logger( 
            verbosity=verbosity,
            print_prefix="[CutManager]"
        )
    
    def add_cut(self, name, description, mask, active=True):
        """
        Add a cut to the collection.
        
        Args: 
            name (str): Name of the cut
            description (str): Description of what the cut does
            mask (awkward.Array): Boolean mask array for the cut
            active (bool, optional): Whether the cut is active by default
        """

        # Get the next available index
        next_idx = len(self.cuts)
    
        self.cuts[name] = {
            "description": description,
            "mask": mask,
            "active": active,
            "idx" : next_idx
        }

        self.logger.log(f"Added cut {name} with index {next_idx}", "info")
        # return self # This would allow method chaining, could be useful maybe?
        
    
    def get_cut_mask(self, name):
        """Utility to return the Boolean mask for a specific cut.
        
        Args:
            name (str): Name of the cut
            
        """
        if name in self.cuts:
            return self.cuts[name]["mask"]
        else:
            self.logger.log(f"Cut '{name}' not defined", "error")
            return None 


    def toggle_cut(self, cut_names, active=False):
        """Utility to set cut(s) as inactive or active 
    
        Args: 
            name (str or list): Name of the cut, or list of cut names
            active (bool, optional): Whether the cut(s) should be active. Default is True.
        """
        # Handle single cut name
        if isinstance(cut_names, str):
            cut_names = [cut_names] 
            
        # Handle list of cut names
        elif isinstance(cut_names, list):
            cut_names = cut_names
        else:
            self.logger.log(f"Invalid input type for cut name(s): {type(cut_names)}", "error")
            return False
        
        # Process each cut name
        success = True
        bad_cuts = []
        
        for cut_name in cut_names:
            if cut_name in self.cuts:
                self.cuts[cut_name]["active"] = active
            else:
                bad_cuts.append(cut_name)
                success = False
        
        # Log results
        if len(bad_cuts) > 0:
            self.logger.log(f"Cut(s) not valid: {bad_cuts}", "error")
        
        if success:
            action = "activated" if active else "deactivated"
            self.logger.log(f"Successfully {action} cut(s): {cut_names}", "info")
        
        return success
    
    def get_active_cuts(self):
        """Utility to get all active cutss"""
        return {name: cut for name, cut in self.cuts.items() if cut["active"]}
    
    def combine_cuts(self, cut_names=None, active_only=True):
        """ Return a Boolean combined mask from specified cuts. Applies an AND operation across all cuts. 
        Args: 

        cut_names (list, optional): List of cut names to include (if None, use all cuts)
        active_only (bool, optional): Whether to only include active cuts
        """

        self.logger.log(f"Combining cuts", "max")
        
        if cut_names is None:
            # Then use all cuts in original order
            cut_names = list(self.cuts.keys())
        # Init mask
        combined = None
        # Loop through cuts        
        for name in cut_names:
            # Get info dict for this cut
            cut_info = self.cuts[name]
            # Active cuts
            if active_only and not cut_info["active"]:
                continue
            # If first cut, initialise 
            if combined is None:
                combined = cut_info["mask"]
            else:
                combined = combined & cut_info["mask"] 
        
        return combined

    def calculate_cut_stats(self, data, progressive=True, active_only=True):
        """ Utility to calculate stats for each cut.
        
        Args:
            data (awkward.Array): Input data 
            progressive (bool, optional): If True, apply cuts progressively; if False, apply each cut independently. Default is True.
            active_only (bool, optional): If True, only include active cuts in statistics
        """
        self.logger.log(f"Calculating cut statistics", "max")
        
        total_events = len(data)
        stats = []
        
        # Base statistics (no cuts)
        stats.append({
            "name": "No cuts",
            "active": "N/A",
            "description": "No selection applied",
            "events_passing": total_events,
            "absolute_frac": 100.0,
            "relative_frac": 100.0
        })
        
        # Get cuts - filter by active status if requested
        if active_only: 
            cuts = [name for name in self.cuts.keys() if self.cuts[name]["active"]]
        else:
            cuts = list(self.cuts.keys())
        
        cumulative_mask = None
        
        for name in cuts:
            cut_info = self.cuts[name]
            mask = cut_info["mask"]
            
            if progressive:
                if cumulative_mask is None:
                    # First cut
                    current_mask = mask
                else:
                    # Apply this cut on top of previous cuts
                    current_mask = cumulative_mask & mask
                cumulative_mask = current_mask
            else:
                # Apply this cut independently
                current_mask = mask
            
            # Calculate event-level efficiency
            event_mask = ak.any(current_mask, axis=-1) # events that have ANY True combined mask
            events_passing = ak.sum(event_mask) # Count up these events
            absolute_frac = events_passing / total_events * 100
            relative_frac = (events_passing / stats[-1]["events_passing"] * 100 
                           if stats[-1]["events_passing"] > 0 else 0)
            
            stats.append({
                "name": name,
                "active": cut_info["active"],
                "description": cut_info["description"],
                "mask": current_mask,
                "events_passing": int(events_passing),
                "absolute_frac": float(absolute_frac),
                "relative_frac": float(relative_frac)
            })
        
        return stats
    
    def print_cut_stats(self, data=None, stats=None, progressive=True, active_only=False, csv_name=None):
        """ Print cut statistics for each cut.
        
        Args:
            data (awkward.Array): Data array
            progressive (bool, optional): If True, apply cuts progressively; if False, apply each cut independently
        """

        self.logger.log(f"Printing cut statistics", "max")
        
        # Input validation
        sources = sum(x is not None for x in [data, stats]) 
        if sources != 1:
            self.logger.log(f"Please provide exactly one of 'data' or 'stats'", "error")
            return None

        if not stats:
            stats = self.calculate_cut_stats(data, progressive, active_only)

        if active_only:
            active_stats = []
            for stat in stats:
                # Always include "No cuts" entry
                if stat["name"] == "No cuts":
                    active_stats.append(stat)
                # Include only active cuts
                elif stat.get("active", True): 
                    active_stats.append(stat)
            stats = active_stats

        # Print header
        self.logger.log(f"Cut statistics", "info")
        print("-" * 110)
        header = "{:<20} {:<20} {:<20} {:<20} {:<30}".format(
            "Cut", "Events passing", "Absolute frac. [%]", "Relative frac. [%]", "Description")
        print(header)
        print("-" * 110)
        # Print each cut's statistics
        for stat in stats:
            row = "{:<20} {:<20} {:<20.2f} {:<20.2f} {:<30}".format(
                stat["name"],
                stat["events_passing"], 
                stat["absolute_frac"], 
                stat["relative_frac"], 
                stat["description"])
            print(row)
        print("-" * 110)
        
        # Print final statistics
        if len(stats) > 1:
            first_events = stats[0]["events_passing"]
            last_events = stats[-1]["events_passing"]
            overall_eff = last_events / first_events * 100 if first_events > 0 else 0
            
            self.logger.log(f"Summary: {last_events}/{first_events} events remaining ({overall_eff:.2f}%)", "info")

        # Write to CSV if requested
        if csv_name:
            try:
                with open(csv_name, 'w', newline='') as csv_file:
                    # Create a CSV writer
                    writer = csv.writer(csv_file)
                    
                    # Write header
                    writer.writerow(['Cut', 'Active', 'Events Passing', 'Absolute Frac. [%]', 'Relative Frac. [%]', 'Description'])
                    
                    # Write data rows
                    for stat in stats:
                        writer.writerow([
                            stat["name"],
                            stat["active"],
                            stat["events_passing"],
                            f"{stat['absolute_frac']:.2f}",
                            f"{stat['relative_frac']:.2f}",
                            stat["description"]
                        ])
                    
                    self.logger.log(f"Cut statistics written to {csv_name}", "success")

            except Exception as e:
                self.logger.log(f"Error writing cut statistics to CSV: {e}", "error")
                
    def save_cuts(self, file_name):
        """ Save the current cut configuration to a JSON file.
        
        Args:
            file_name (str): File path to save the configuration
        """
        config = {
            "cuts": {}
        }
        
        for name, info in self.cuts.items():
            config["cuts"][name] = {
                "description": info["description"],
                "active": info["active"],
                "idx": info["idx"]
            }
        
        with open(file_name, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.log(f"Saved cut configuration to {file_name}", "success")

    def combine_cut_stats(self, stats_list, active_only=True):
        """Combine a list of cut statistics after multiprocessing 
        
        Args:
            stats_list: List of cut statistics lists from different files
            active_only (bool, optional): If True, only include active cuts in combined stats
        
        Returns:
            list: Combined cut statistics
        """
        self.logger.log(f"Combining cut statistics", "max")
        
        # Return empty list if no input
        if not stats_list:
            return []
        
        # Filter ALL stats lists based on active_only flag, not just the template
        if active_only:
            filtered_stats_list = []
            for stats in stats_list:
                filtered_stats = []
                for cut in stats:
                    # Always include "No cuts" entry
                    if cut["name"] == "No cuts":
                        filtered_stats.append(cut)
                    # Include only active cuts
                    elif cut.get("active", True):
                        filtered_stats.append(cut)
                filtered_stats_list.append(filtered_stats)
            stats_list = filtered_stats_list
        
        # Use the first (now filtered) list as template
        template_stats = stats_list[0]
        
        # Use the template to initialize combined stats
        combined_stats = []
        for cut in template_stats:
            # Create a copy without the mask (which we don't need)
            cut_copy = {k: v for k, v in cut.items() if k != 'mask'}
            # Reset the event count
            cut_copy['events_passing'] = 0
            combined_stats.append(cut_copy)
        
        # Create a mapping of cut names to indices in combined_stats 
        cut_name_to_index = {cut['name']: i for i, cut in enumerate(combined_stats)}
        
        # Sum up events_passing for each cut across all files
        for stats in stats_list:  # Now this is filtered!
            for cut in stats:
                cut_name = cut['name']
                # Only process cuts that are in our combined_stats
                if cut_name in cut_name_to_index:
                    idx = cut_name_to_index[cut_name]
                    combined_stats[idx]['events_passing'] += cut['events_passing']
        
        # Recalculate percentages
        if combined_stats and combined_stats[0]['events_passing'] > 0:
            total_events = combined_stats[0]['events_passing']
            
            for i, cut in enumerate(combined_stats):
                events = cut['events_passing']
                
                # Absolute percentage
                cut['absolute_frac'] = (events / total_events) * 100.0
                
                # Relative percentage
                if i == 0:  # "No cuts"
                    cut['relative_frac'] = 100.0
                else:
                    prev_events = combined_stats[i-1]['events_passing']
                    cut['relative_frac'] = (events / prev_events) * 100.0 if prev_events > 0 else 0.0
        
        return combined_stats
