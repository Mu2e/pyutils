# Sam Grant
# May 2025
# Tests for pyutils modules 

# For safety:
# Use the 'spawn' start method to ensure each worker process starts in a fresh Python interpreter.
# This avoids inheriting state from the parent (as happens with 'fork'), which can cause subtle bugs
# with threads, open files, or shared resources. Slightly slower, but much safer for complex code.
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# pyutils classes
from pyutils.pyread import Reader                  # Data reading 
from pyutils.pyprocess import Processor, Skeleton  # Data processing
from pyutils.pyimport import Importer              # TTree (EventNtuple) importing 
from pyutils.pyplot import Plot                    # Plotting and visualisation 
from pyutils.pyprint import Print                  # Array visualisation 
from pyutils.pyselect import Select                # Data selection and cut management 
from pyutils.pyvector import Vector                # Element wise vector operations
from pyutils.pylogger import Logger                # Printout manager

import gc

# Cannot be nested (for multiprocessing)!
class MyProcessor(Skeleton):
    def __init__(self, file_list_path, use_processes):
        super().__init__()
        self.file_list_path=file_list_path 
        self.use_processes=True
    def process_file(self, file_name):
        return file_name 
            
class Tester:
    """ Tests for pyutils """
    
    def __init__(self):  
        # Max verbosity 
        self.verbosity = 2 
        
        # Error tracking
        self.error_count = 0
        self.test_count = 0
        self.failed_tests = []

        # Test files
        self.local_file_path = "/exp/mu2e/data/users/sgrant/pyutils-test/files/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000699.root"
        self.local_wb_file_path = "/exp/mu2e/data/users/sgrant/pyutils-test/files/rec.mu2e.CRV_wideband_cosmics.CRVWB-000-012-000.000105_001.root"
        self.remote_file_name = "nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000699.root"
        self.remote_wideband_file_name = "rec.mu2e.CRV_wideband_cosmics.CRVWB-000-012-000.000105_001.root" 
        self.local_file_list = "/exp/mu2e/data/users/sgrant/pyutils-test/filelists/local_file_list.txt"
        self.bad_local_file_list = "/exp/mu2e/data/users/sgrant/pyutils-test/filelists/bad_local_file_list.txt"
        self.remote_file_list = "/exp/mu2e/data/users/sgrant/pyutils-test/filelists/remote_file_list.txt"
        self.defname = "nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.root"
        
        # Setup logger 
        self.logger = Logger(
            print_prefix="[pytest]", 
            verbosity=2  # max verbosity 
        )
        self.logger.log("Initialised", "success")
    
    def _safe_test(self, test_name, test_function, *args, expect_return=True, **kwargs):
        """Wrapper to safely run tests and count errors"""
        self.test_count += 1
        try:
            self.logger.log(f"Running test: {test_name}", "test")
            result = test_function(*args, **kwargs)            
            if expect_return and (result is None or (hasattr(result, '__len__') and len(result) == 0)):
                self.logger.log(f"FAILED: {test_name}: returned no results", "error")
                self.error_count += 1
                self.failed_tests.append(test_name)
                return False
            else:
                self.logger.log(f"PASSED: {test_name}", "success")
                return True
        except Exception as e:
            self.logger.log(f"FAILED: {e}", "error")
            self.error_count += 1
            self.failed_tests.append(test_name)
            return False
        finally:
            gc.collect()
            del result
    
    ###### pyread ######
    
    def _local_read(self):  
        self.logger.log("Local read", "test")  
        # Start reader
        reader = Reader(verbosity=self.verbosity)
        # Read
        return reader.read_file(file_path=self.local_file_path)  
    
    def _remote_read(self):  
        self.logger.log("Remote read", "test")  
        # Start reader
        reader = Reader(
            use_remote=True,
            location="tape",
            verbosity=self.verbosity
        )
        # Read
        return reader.read_file(file_path=self.remote_file_name)  
    
    def _test_reader(self, local_read=True, remote_read=True): 
        """Test pyread:Reader module"""
        self.logger.log("Testing pyread:Reader", "test")  
        
        if local_read: 
            self._safe_test("pyread:Reader:read_file (local)", self._local_read)
        
        if remote_read: 
            self._safe_test("pyread:Reader::read_file (remote)", self._remote_read)

    ###### pyimport ######

    def _local_import_branch(self):
        importer = Importer(
            file_name = self.local_file_path,
            branches = ["event"],
            use_remote=False,
            verbosity = self.verbosity
        )

        return importer.import_branches()

    def _local_import_wb_branch(self): # Wideband (WB) tree 
        importer = Importer(
            file_name = self.local_wb_file_path,
            tree_path = "run",
            branches = ["runNumber"],
            use_remote=False,
            verbosity = self.verbosity
        )

        return importer.import_branches()

    def _local_import_special_branch(self): # Branches with special characters in the name
        importer = Importer(
            file_name = self.local_file_path,
            branches = ["crvcoincs.PEsPerLayer[4]", "crvcoincs.sidePEsPerLayer[8]"],
            use_remote=False,
            verbosity = self.verbosity
        )

        return importer.import_branches()
    
    def _remote_import_branch(self):
        importer = Importer(
            file_name = self.local_file_path,
            branches = ["event"],
            use_remote=True,
            location="tape",
            verbosity = self.verbosity
        )

        return importer.import_branches()

    def _local_import_grouped_branches(self):
        importer = Importer(
            file_name = self.local_file_path,
            branches = {
                "evt" : ["event"],
                "crv" : ["crvcoincs.PEs", "crvcoincs.nHits"]
            },
            use_remote=False,
            verbosity = self.verbosity
        )

        return importer.import_branches()
        
    def _local_import_all_branches(self):
        importer = Importer(
            file_name = self.local_file_path,
            branches = "*",
            use_remote=False,
            verbosity = 1 # this one is loud! 
        )

        return importer.import_branches()
    
    def _test_importer(
        self, 
        local_import_branch=True,
        local_import_wb_branch=True,
        local_import_special_branch=True,
        remote_import_branch=True,
        local_import_grouped_branches=True,
        local_import_all_branches=True 
    ):
        """Test pyimport:Importer module"""
        self.logger.log("Testing pyimport:Importer", "info")  

        if local_import_branch:
            self._safe_test("pyimport:Importer:import_branches (local, single branch)", self._local_import_branch)

        if local_import_wb_branch:
            self._safe_test("pyimport:Importer:import_branches (local, single WB file, single branch)", self._local_import_wb_branch)            

        if local_import_special_branch:
            self._safe_test("pyimport:Importer:import_branches (local, single file, special branches)", self._local_import_special_branch)     
            
        if local_import_branch:
            self._safe_test("pyimport:Importer:import_branches (remote, single branch)", self._remote_import_branch)
            
        if local_import_grouped_branches:
            self._safe_test("pyimport:Importer:import_branches (local, grouped branches)", self._local_import_grouped_branches)

        if local_import_all_branches:
            self._safe_test("pyimport:Importer:import_branches (local, all branches)", self._local_import_all_branches)

        # if remote_wideband_import_branch:
        #     self._safe_test("pyimport:Importer:import_branches (remote, wideband, single branch)", self._remote_wideband_import_branch)
            
    ###### pyprocess ######
    
    def _local_process_file(self):
        processor = Processor(verbosity=self.verbosity)
        return processor.process_data(
            file_name=self.local_file_path,
            branches=["event"]
        )

    def _local_process_file_wrong_loc(self):
        processor = Processor(
            use_remote=False,
            verbosity=self.verbosity,
            location="scratch" # INCORRECT
        )
        return processor.process_data(
            file_name=self.local_file_path,
            branches=["event"]
        )

    def _local_process_wb_file(self): # WB tree structure
        processor = Processor(
            tree_path="run",
            verbosity=self.verbosity
        )
        return processor.process_data(
            file_name=self.local_wb_file_path,
            branches=["runNumber"]
        )

    def _local_process_file_special_branch(self): # Branches with special characters in the name
        processor = Processor(verbosity=self.verbosity)
        return processor.process_data(
            file_name=self.local_file_path,
            branches=["crvcoincs.PEsPerLayer[4]", "crvcoincs.sidePEsPerLayer[8]"]
        )

    def _remote_process_file(self):
        processor = Processor(
            use_remote=True,
            verbosity=self.verbosity
        )
        return processor.process_data(
            file_name=self.remote_file_name,
            branches=["event"]
        )

    def _remote_process_file(self):
        processor = Processor(
            use_remote=True,
            verbosity=self.verbosity
        )
        return processor.process_data(
            file_name=self.remote_file_name,
            branches=["event"]
        )

    def _remote_process_file_wrong_loc(self):
        processor = Processor(
            use_remote=True,
            verbosity=self.verbosity,
            location="scratch" # INCORRECT
        )
        return processor.process_data(
            file_name=self.remote_file_name,
            branches=["event"]
        )
        
    def _local_get_file_list(self):
        processor = Processor(verbosity=self.verbosity)
        return processor.get_file_list(file_list_path=self.local_file_list)

    def _remote_get_file_list(self):
        processor = Processor(
            use_remote=True,
            verbosity=self.verbosity
        )
        return processor.get_file_list(file_list_path=self.remote_file_list)

    def _sam_get_file_list(self):
        processor = Processor(
            use_remote=True,
            verbosity=self.verbosity
        )
        return processor.get_file_list(defname=self.defname)

    # def _sam_get_file_list_TEST(self):
    #     processor = Processor(
    #         use_remote=False,
    #         verbosity=self.verbosity
    #     )
    #     return processor.get_file_list(defname=self.defname)

    def _basic_multithread(self):
        processor = Processor(
            verbosity=self.verbosity
        )
        return processor.process_data(
            file_list_path=self.local_file_list,
            branches = ["event"]
        )

    def _basic_remote_multithread(self):
        processor = Processor(
            location="disk",
            verbosity=self.verbosity,
            use_remote=True
        )
        return processor.process_data(
            file_list_path=self.remote_file_list,
            branches = ["event"]
        )

    def _basic_bad_multithread(self):
        processor = Processor(
            verbosity=self.verbosity
        )
        return processor.process_data(
            file_list_path=self.bad_local_file_list,
            branches = ["event"]
        )
        
    def _basic_multiprocess(self):
        processor = Processor(
            verbosity=self.verbosity, 
            worker_verbosity=2
        )
        return processor.process_data(
            file_list_path=self.local_file_list,
            branches = ["event"],
            use_processes=True
        )

    def _basic_remote_multiprocess(self):
        processor = Processor(
            verbosity=self.verbosity,
            use_remote=True,
            worker_verbosity=2,
            location="disk"
        )
        return processor.process_data(
            file_list_path=self.remote_file_list,
            branches = ["event"],
            use_processes=True
        )
            
    def _advanced_multithread(self):
        my_processor = MyProcessor(self.local_file_list, False)
        return my_processor.execute()

    def _advanced_multiprocess(self):
        my_processor = MyProcessor(self.local_file_list, True)
        return my_processor.execute()

    def _test_processor(
        self, 
        local_process_file=True,
        local_process_wb_file=True,
        local_process_file_special_branch=True,
        remote_process_file=True,
        get_file_list=True,
        basic_multifile=True,
        advanced_multifile=True
    ):
        """Test pyprocess module"""
        if local_process_file:
            self._safe_test("pyprocess:Processor:process_data (local, single file, single branch)", self._local_process_file)
            # self._safe_test("pyprocess:Processor:process_data (local, single file, single branch, wrong file location)", self._local_process_file_wrong_loc)

        if local_process_wb_file:
            self._safe_test("pyprocess:Processor:process_data (local, single WB file, single branch)", self._local_process_wb_file)            

        if local_process_file_special_branch:
            self._safe_test("pyprocess:Processor:process_data (local, single file, special branches)", self._local_process_file_special_branch)     
            
        if remote_process_file:
            self._safe_test("pyprocess:Processor:process_data (remote, single file, single branch)", self._remote_process_file)
            # self._safe_test("pyprocess:Processor:process_data (remote, single file, single branch, wrong file location)", self._remote_process_file_wrong_loc)

        if get_file_list:
            self._safe_test("pyprocess:Processor:get_file_list (local file list path)", self._local_get_file_list)
            self._safe_test("pyprocess:Processor:get_file_list (remote file list path)", self._remote_get_file_list)
            self._safe_test("pyprocess:Processor:get_file_list (SAM definition)", self._sam_get_file_list)
            # self._safe_test("pyprocess:Processor:get_file_list (SAM definition)", self._sam_get_file_list_TEST)

        if basic_multifile:
            self._safe_test("pyprocess:Processor:process_data (basic multithread)", self._basic_multithread)
            self._safe_test("pyprocess:Processor:process_data (basic remote multithread)", self._basic_remote_multithread)
            # self._safe_test("pyprocess:Processor:process_data (basic bad multithread)", self._basic_bad_multithread)
            self._safe_test("pyprocess:Processor:process_data (basic multiprocess)", self._basic_multiprocess)
            self._safe_test("pyprocess:Processor:process_data (basic remote multiprocess)", self._basic_remote_multiprocess)
            # self._safe_test("pyprocess:Processor:process_data (basic remote multithread)", self._basic_remote_multiprocess)

        if advanced_multifile:
            self._safe_test("pyprocess:Skeleton (advanced multithread)", self._advanced_multithread)
            self._safe_test("pyprocess:Skeleton (advanced multiprocess)", self._advanced_multiprocess)

    ###### pyselect ######

    def _is_electron(self, selector, data):
        return selector.is_electron(data)
        
    def _is_positron(self, selector, data):
        return selector.is_positron(data)

    def _is_mu_minus(self, selector, data):
        return selector.is_mu_minus(data)

    def _is_mu_plus(self, selector, data):
        return selector.is_mu_plus(data)

    def _is_particle(self, selector, data):
        return selector.is_particle(data, "e-") 
        
    def _is_upstream(self, selector, data):
        return selector.is_upstream(data) 

    def _is_downstream(self, selector, data):
        return selector.is_downstream(data) 

    def _is_reflected(self, selector, data):
        return selector.is_reflected(data) 

    def _select_trkqual(self, selector, data):
        return selector.select_trkqual(data, quality=0.5) 

    def _has_n_hits(self, selector, data):
        return selector.has_n_hits(data, n_hits=1) 
        
    def _test_select(
        self,
        particle_masks=True,
        momentum_masks=True,
        trk_masks=True
    ):
        # Get data
        processor = Processor(verbosity=0)
        data = processor.process_data(
            file_name=self.local_file_path,
            branches={
                "v" : [ "trk.pdg", "trk.nactive", "trkqual.result"], # vectors
                "vov" : [ "trksegs" ] # vector of vectors
            }
        )
        # Get selector
        selector = Select(verbosity=self.verbosity)

        if particle_masks:
            self._safe_test("pyselect:Selector:is_electron (local, single file)", self._is_electron, selector, data["v"])
            self._safe_test("pyselect:Selector:is_positron (local, single file)", self._is_positron, selector, data["v"])
            self._safe_test("pyselect:Selector:is_mu_minus (local, single file)", self._is_mu_minus, selector, data["v"])
            self._safe_test("pyselect:Selector:is_mu_plus (local, single file)", self._is_mu_plus, selector, data["v"])
            self._safe_test("pyselect:Selector:is_particle (local, single file)", self._is_particle, selector, data["v"])

        if momentum_masks:
            self._safe_test("pyselect:Selector:is_upstream (local, single file)", self._is_upstream, selector, data["vov"])
            self._safe_test("pyselect:Selector:is_downstream (local, single file)", self._is_downstream, selector, data["vov"])
            self._safe_test("pyselect:Selector:is_reflected (local, single file)", self._is_reflected, selector, data["vov"])

        if trk_masks: 
            self._safe_test("pyselect:Selector:select_trkqual (local, single file)", self._select_trkqual, selector, data["v"])
            self._safe_test("pyselect:Selector:has_n_hits (local, single file)", self._has_n_hits, selector, data["v"])
            
    ###### pyprint ######   
    
    def _normal_print(self, data):
        printer = Print()
        return printer.print_n_events(data)
        
    def _verbose_print(self, data):
        printer = Print(verbose=True)
        return printer.print_n_events(data)

    def _test_print(
        self,
        normal_print=True,
        verbose_print=True
    ):
        # Get data
        processor = Processor(verbosity=self.verbosity)
        data = processor.process_data(
            file_name=self.local_file_path,
            branches=["event", "crvcoincs.PEs", "trk.pdg", "trksegs"]
        )
        if normal_print:
            self._safe_test("pyprint:Print:print_n_events (local, single file, verbose=False)", self._normal_print, data, expect_return=False)
        if verbose_print:
            self._safe_test("pyprint:Print:print_n_events (local, single file, verbose=False)", self._verbose_print, data, expect_return=False)

    ###### pyvector ######

    def _get_vector(self, vector, branch, vector_name):
        return vector.get_vector(branch, vector_name) 

    def _get_mag(self, vector, branch, vector_name):
        return vector.get_mag(branch, vector_name) 
        
    def _test_vector(
        self,
        get_vector=True,
        get_mag=True
    ):
        # Get data
        processor = Processor(verbosity=0) 
        data = processor.process_data(
            file_name=self.local_file_path,
            branches=["trksegs"]
        )
        # Get selector
        vector = Vector(verbosity=self.verbosity)

        if get_vector:
            self._safe_test("pyvector:Vector:get_vector (local, single file, trksegs, mom)", self._get_vector, vector, data["trksegs"], "mom") 
            self._safe_test("pyvector:Vector:get_vector (local, single file, trksegs, pos)", self._get_vector, vector, data["trksegs"], "pos") 

        if get_mag: 
            self._safe_test("pyvector:Vector:get_vector (local, single file, trksegs, mom)", self._get_mag, vector, data["trksegs"], "mom") 
            self._safe_test("pyvector:Vector:get_vector (local, single file, trksegs, pos)", self._get_mag, vector, data["trksegs"], "pos") 
            
    ####### TODO: Add more test methods for plot ######
    
    def _test_plot(self):
        plotter = Plot()
    
    ###### Test summary ######

    def print_summary(self):
       """Print test summary"""
       failed_tests_str = "\n".join([f"  - {test}" for test in self.failed_tests]) if self.failed_tests else ""
       final_status = "🎉 All tests passed!" if self.error_count == 0 else f"⚠️ {self.error_count} test(s) failed"
       
       summary = f"""
                {'=' * 50}
                TEST SUMMARY
                {'=' * 50}
                Total tests run: {self.test_count}
                Passed: {self.test_count - self.error_count}
                Failed: {self.error_count}
                {f'Failed tests:\n{failed_tests_str}' if self.failed_tests else ''}
                {final_status}"""
       
       self.logger.log(summary, "test")
    
    ###### run tests ######
    
    def run(
        self, 
        test_reader=False,
        test_processor=False, 
        test_importer=False,
        test_select=False,
        test_plot=False,
        test_print=False,
        test_vector=False
        ): 
        """Run all specified tests"""
        
        if test_reader:
            self.logger.log("************ Testing pyread ************", "test")
            self._test_reader()

        if test_importer:
            self.logger.log("************ Testing pyimport ************", "test")
            self._test_importer()

        if test_processor:
            self.logger.log("************ Testing pyprocess ************", "test")
            self._test_processor()            
    
        if test_plot:
            self.logger.log("************ Testing pyplot ************", "test")
            self._test_plot()
            
        if test_print:
            self.logger.log("************ Testing pyprint ************", "test")
            self._test_print()
            
        if test_select:
            self.logger.log("************ Testing pyselect ************", "test")
            self._test_select()
            
        if test_vector:
            self.logger.log("************ Testing pyvector ************", "test")
            self._test_vector()
        
        # Print final summary
        self.print_summary()
        
        return self.error_count == 0  # Return True if all tests passed

# Usage (see run_tests.ipynb)
# tester = Tester()
# tester.run()  # Run all tests

# Or run specific tests:
# tester.run(test_reader=True, test_processor=False)