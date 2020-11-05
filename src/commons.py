

def make_deterministic(seed=0):
    """
    Rende deterministici i risultati. 
    Nota che per alcune librerie (es: la PCA di sklearn) ciò non è sufficiente.
    """
    import torch
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True   # questo rallenta (dicono)
    torch.backends.cudnn.benchmark = False      # questo rallenta (dicono)


def setup_logging(output_folder, exist_ok=False, console="debug",
                  info_filename="info.log", debug_filename="debug.log"):
    """Set up logging files and console output.
    Creates one file for INFO logs and one for DEBUG logs.
    Args:
        output_folder (str): creates the folder where to save the files.
        exist_ok (boolean): if False throw a FileExistsError if output_folder already exists
        debug (str):
            if == "debug" prints on console debug messages and higher
            if == "info"  prints on console info messages and higher
            if == None does not use console (useful when a logger has already been set)
        info_filename (str): the name of the info file. if None, don't create info file
        debug_filename (str): the name of the debug file. if None, don't create debug file
    """
    import os
    import sys
    import logging
    import traceback
    if not exist_ok and os.path.exists(output_folder):
        raise FileExistsError(f"{output_folder} esiste già !!!")
    os.makedirs(output_folder, exist_ok=True)
    logging.getLogger("matplotlib.font_manager").disabled = True
    base_formatter = logging.Formatter("%(asctime)s   %(message)s", "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    
    if info_filename != None:
        info_file_handler = logging.FileHandler(f"{output_folder}/{info_filename}")
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)
    
    if debug_filename != None:
        debug_file_handler = logging.FileHandler(f"{output_folder}/{debug_filename}")
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)
    
    if console != None:
        console_handler = logging.StreamHandler()
        if console == "debug": console_handler.setLevel(logging.DEBUG)
        if console == "info":  console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)
    
    def my_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = my_handler

