import json
import awkward as ak
import csv
import pandas as pd
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
        # Store original active states for restoration
        self._original_states = {}
        # Start logger
        self.logger = Logger( 
            verbosity=verbosity,
            print_prefix="[pycut]"
        )
        self.logger.log("Initialised", "success")
    
    def add_cut(self, name, description, mask, active=True, group=None):
        """
        Add a cut to the collection.
        
        Args: 
            name (str): Name of the cut
            description (str): Description of what the cut does
            mask (awkward.Array): Boolean mask array for the cut
            active (bool, optional): Whether the cut is active by default
            group (str, optional): Group name for organizing cuts
        """

        # Get the next available index
        next_idx = len(self.cuts)
    
        self.cuts[name] = {
            "description": description,
            "mask": mask,
            "active": active,
            "group": group,
            "idx" : next_idx
        }

        # Store original state for restoration
        self._original_states[name] = active

        group_info = f" in group '{group}'" if group else ""
        self.logger.log(f"Added cut {name} with index {next_idx}{group_info}", "info")
        # return self # This would allow method chaining, could be useful maybe?

    def toggle_cut(self, cut_dict):
        """Utility to set cut(s) as inactive or active based on input dictionary
        
        Args: 
            cut_dict (dict): Dictionary mapping cut names to their desired active state
                            e.g., {"cut_name_1": False, "cut_name_2": True}
        """
        # Validate input type
        if not isinstance(cut_dict, dict):
            self.logger.log(f"Invalid input type: expected dict, got {type(cut_dict)}", "error")
            return False
        
        # Process each cut name and state
        success = True
        bad_cuts = []
        activated_cuts = []
        deactivated_cuts = []
        
        for cut_name, active_state in cut_dict.items():
            if cut_name in self.cuts:
                self.cuts[cut_name]["active"] = active_state
                if active_state:
                    activated_cuts.append(cut_name)
                else:
                    deactivated_cuts.append(cut_name)
            else:
                bad_cuts.append(cut_name)
                success = False
        
        # Log results
        if len(bad_cuts) > 0:
            self.logger.log(f"Cut(s) not valid: {bad_cuts}", "error")
        
        if len(activated_cuts) > 0:
            self.logger.log(f"Successfully activated cut(s): {activated_cuts}", "info")
        
        if len(deactivated_cuts) > 0:
            self.logger.log(f"Successfully deactivated cut(s): {deactivated_cuts}", "info")
        
        return success

    def toggle_group(self, group_dict):
        """Utility to set entire group(s) of cuts as inactive or active
        
        Args: 
            group_dict (dict): Dictionary mapping group names to their desired active state
                              e.g., {"quality_cuts": False, "momentum_cuts": True}
        """
        # Validate input type
        if not isinstance(group_dict, dict):
            self.logger.log(f"Invalid input type: expected dict, got {type(group_dict)}", "error")
            return False
        
        success = True
        bad_groups = []
        activated_groups = []
        deactivated_groups = []
        
        for group_name, active_state in group_dict.items():
            # Find all cuts in this group
            cuts_in_group = [name for name, cut_info in self.cuts.items() 
                           if cut_info.get("group") == group_name]
            
            if cuts_in_group:
                # Toggle all cuts in the group
                for cut_name in cuts_in_group:
                    self.cuts[cut_name]["active"] = active_state
                
                if active_state:
                    activated_groups.append(f"{group_name} ({len(cuts_in_group)} cuts)")
                else:
                    deactivated_groups.append(f"{group_name} ({len(cuts_in_group)} cuts)")
            else:
                bad_groups.append(group_name)
                success = False
        
        # Log results
        if len(bad_groups) > 0:
            self.logger.log(f"Group(s) not found: {bad_groups}", "error")
        
        if len(activated_groups) > 0:
            self.logger.log(f"Successfully activated group(s): {activated_groups}", "info")
        
        if len(deactivated_groups) > 0:
            self.logger.log(f"Successfully deactivated group(s): {deactivated_groups}", "info")
        
        return success

    def save_state(self, state_name="default"):
        """Save current active states of all cuts
        
        Args:
            state_name (str): Name for this saved state
        """
        if not hasattr(self, '_saved_states'):
            self._saved_states = {}
            
        self._saved_states[state_name] = {
            name: cut_info["active"] for name, cut_info in self.cuts.items()
        }
        self.logger.log(f"Saved current cut states as '{state_name}'", "info")

    def restore_state(self, state_name="default"):
        """Restore previously saved cut states
        
        Args:
            state_name (str): Name of the saved state to restore
        """
        if not hasattr(self, '_saved_states') or state_name not in self._saved_states:
            self.logger.log(f"No saved state '{state_name}' found", "error")
            return False
            
        saved_states = self._saved_states[state_name]
        restored_cuts = []
        
        for cut_name, active_state in saved_states.items():
            if cut_name in self.cuts:
                self.cuts[cut_name]["active"] = active_state
                restored_cuts.append(cut_name)
            else:
                self.logger.log(f"Cut '{cut_name}' no longer exists, skipping", "warning")
        
        self.logger.log(f"Restored {len(restored_cuts)} cuts from state '{state_name}'", "success")
        return True

    def restore_original_state(self):
        """Restore all cuts to their original active states (as defined when added)"""
        restored_cuts = []
        
        for cut_name, original_state in self._original_states.items():
            if cut_name in self.cuts:
                self.cuts[cut_name]["active"] = original_state
                restored_cuts.append(cut_name)
            else:
                self.logger.log(f"Cut '{cut_name}' no longer exists, skipping", "warning")
        
        self.logger.log(f"Restored {len(restored_cuts)} cuts to original states", "success")
        return True

    def list_saved_states(self):
        """List all saved states"""
        if not hasattr(self, '_saved_states') or not self._saved_states:
            self.logger.log("No saved states found", "info")
            return []
            
        states = list(self._saved_states.keys())
        self.logger.log(f"Available saved states: {states}", "info")
        return states

    def get_groups(self):
        """Get all unique group names and their cuts
        
        Returns:
            dict: Dictionary mapping group names to lists of cut names
        """
        groups = {}
        for cut_name, cut_info in self.cuts.items():
            group = cut_info.get("group")
            if group is not None:
                if group not in groups:
                    groups[group] = []
                groups[group].append(cut_name)
        
        # Also add ungrouped cuts
        ungrouped = [cut_name for cut_name, cut_info in self.cuts.items() 
                    if cut_info.get("group") is None]
        if ungrouped:
            groups[None] = ungrouped
            
        return groups

    def list_groups(self):
        """Print all groups and their cuts"""
        groups = self.get_groups()
        
        if not groups:
            self.logger.log("No cuts defined", "info")
            return
            
        for group_name, cut_names in groups.items():
            if group_name is None:
                self.logger.log(f"Ungrouped cuts ({len(cut_names)}): {', '.join(cut_names)}", "info")
            else:
                active_cuts = [name for name in cut_names if self.cuts[name]["active"]]
                self.logger.log(f"Group '{group_name}' ({len(active_cuts)}/{len(cut_names)} active): {', '.join(cut_names)}", "info")
    
    def get_active_cuts(self):
        """Utility to get all active cutss"""
        return {name: cut for name, cut in self.cuts.items() if cut["active"]}
    
    def combine_cuts(self, cut_names=None, active_only=True):
        """ Return a Boolean combined mask from specified cuts. Applies an AND operation across all cuts. 
        Args: 

        cut_names (list, optional): List of cut names to include (if None, use all cuts)
        active_only (bool, optional): Whether to only include active cuts
        """        
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

    ############################################
    # Generate and manage cut flows
    ############################################
    
    def _add_entry(self, name, events_passing, absolute_frac, relative_frac, description, group=None):
        entry = {
            "name": name,
            "events_passing": int(events_passing),
            "absolute_frac": round(float(absolute_frac), 2),
            "relative_frac": round(float(relative_frac), 2),
            "description": description
        }
        if group is not None:
            entry["group"] = group
        return entry
            
    def create_cut_flow(self, data):
        """ Utility to calculate cut flow from array and cuts object
        
        Args:
            data (awkward.Array): Input data 
        """
        total_events = len(data)
        cut_flow = []
        
        # Base statistics (no cuts)
        cut_flow.append(
            self._add_entry(
                name = "No cuts",
                group = "N/A",
                events_passing = total_events,
                absolute_frac =  100.00,
                relative_frac = 100.00,
                description = "No selection applied",
            )
        )
        
        # Get cuts, filter by active status
        cuts = [name for name in self.cuts.keys() if self.cuts[name]["active"]]

        # Initialise cumulative mask
        cumulative_mask = None
        
        for name in cuts:
            # Get info for this cut
            cut_info = self.cuts[name]
            # Get mask for this cut
            mask = cut_info["mask"]

            # Combine cuts progressively
            if cumulative_mask is None:
                # First cut
                current_mask = mask
            else:
                # Apply this cut on top of previous cuts
                current_mask = cumulative_mask & mask
                
            # Redefine cumulative mask    
            cumulative_mask = current_mask

            # Calculate event-level efficiency
            event_mask = ak.any(current_mask, axis=-1) # events that have ANY True combined mask
            events_passing = ak.sum(event_mask) # Count up these events
            absolute_frac = events_passing / total_events * 100
            relative_frac = (events_passing / cut_flow[-1]["events_passing"] * 100 
                           if cut_flow[-1]["events_passing"] > 0 else 0)

            # Append row
            cut_flow.append(
                self._add_entry(
                    name = name,
                    events_passing = events_passing,
                    absolute_frac = absolute_frac,
                    relative_frac = relative_frac,
                    description = cut_info["description"],
                    group = cut_info.get("group")
                )
            )

        return cut_flow

    def format_cut_flow(self, cut_flow, include_group=True):
        """Format cut flow as a DataFrame with more readable column names

            Args:
                cut_flow (dict): The cut flow to format
                include_group (bool, optional): Whether to include group column
            Returns:
                df_cut_flow (pd.DataFrame)
        """
        df_cut_flow = pd.DataFrame(cut_flow)
        
        column_mapping = {
            "name": "Cut",
            "events_passing": "Events Passing",
            "absolute_frac": "Absolute [%]", 
            "relative_frac": "Relative [%]",
            "description": "Description"
        }
        
        if include_group and "group" in df_cut_flow.columns:
            column_mapping["group"] = "Group"
        
        df_cut_flow = df_cut_flow.rename(columns=column_mapping)
        
        # Reorder columns to put Group after Cut if it exists
        if include_group and "Group" in df_cut_flow.columns:
            cols = df_cut_flow.columns.tolist()
            if "Group" in cols:
                cols.remove("Group")
                cut_idx = cols.index("Cut") if "Cut" in cols else 0
                cols.insert(cut_idx + 1, "Group")
                df_cut_flow = df_cut_flow[cols]
        
        return df_cut_flow
        
    def combine_cut_flows(self, cut_flow_list, format_as_df=True):
        """Combine a list of cut flows after multiprocessing 
        
        Args:
            cut_flows: List of cut statistics lists from different files
            format_as_df (bool, optional): Format output as a pd.DataFrame. Defaults to True.
        
        Returns:
            list: Combined cut statistics
        """        
        # Return empty list if no input
        if not cut_flow_list:
            self.logger.log(f"No cut flows to combine", "error")
            return []

        try:
            # Use the first (now filtered) list as template
            template = cut_flow_list[0]
            
            # Use the template to initialise combined stats
            combined_cut_flow = []
            for cut in template:
                # Create a copy (needed?)
                cut_copy = {k: v for k, v in cut.items()}
                # Reset the event count
                cut_copy["events_passing"] = 0
                combined_cut_flow.append(cut_copy)
            
            # Create a mapping of cut names to indices in combined_stats 
            cut_name_to_index = {cut["name"]: i for i, cut in enumerate(combined_cut_flow)}
            
            # Sum up events_passing for each cut across all files
            for cut_flow in cut_flow_list:
                for cut in cut_flow:
                    cut_name = cut["name"]
                    # Only process cuts that are in our combined_stats
                    if cut_name in cut_name_to_index:
                        idx = cut_name_to_index[cut_name]
                        combined_cut_flow[idx]["events_passing"] += cut["events_passing"]
            
            # Recalculate percentages
            if combined_cut_flow and combined_cut_flow[0]["events_passing"] > 0:
                total_events = combined_cut_flow[0]["events_passing"]
                
                for i, cut in enumerate(combined_cut_flow):
                    events = cut["events_passing"]
                    
                    # Absolute percentage
                    cut["absolute_frac"] = (events / total_events) * 100.0
                    
                    # Relative percentage
                    if i == 0:  # "No cuts"
                        cut["relative_frac"] = 100.0
                    else:
                        prev_events = combined_cut_flow[i-1]["events_passing"]
                        cut["relative_frac"] = (events / prev_events) * 100.0 if prev_events > 0 else 0.0
    
            if format_as_df:
                self.logger.log(f"Combined and formatted cut flows", "success")
                return self.format_cut_flow(combined_cut_flow)
            else:
                self.logger.log(f"Combined cut flows", "success")
                return combined_cut_flow
        
        except Exception as e:
            self.logger.log(f"Exception when combining cut flows: {e}", "error")
            raise