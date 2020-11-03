

def pause_while_running(pid):
    """Se il PID è -1 parte subito"""
    import psutil
    import time
    import os
    print(f"Sono il processo {os.getpid()} e sto aspettando il processo {pid} ...")
    while int(pid) in psutil.pids():
        time.sleep(5)
    print(f"Ora parto")


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


def create_dir_if_not_exists(dir_name):
    """Crea una cartella (o albero di cartelle) se non esiste già"""
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


class Logger:
    def __init__(self, folder=".", filename="logger.txt", 
                 print_day=False, print_time=True):
        """
        Crea un file e ci scrive dentro quello che viene passato alla funzione
        log(line, print_in_file=True). 
        
        Esempi:
            if print_day:
                line = '2020-03-29 17:11:46' + line
            elif print_time:
                line = '17:11:51' + line
        """
        import os
        import sys
        self.log_filename = os.path.join(folder, filename)
        self.full_log_filename = os.path.join(folder, "." + filename)
        self.print_day = print_day
        self.print_time = print_time
        
        # creo folder e files se non esistono già
        create_dir_if_not_exists(folder)
        if not os.path.exists(self.log_filename):
            open(self.log_filename, 'a').close()
        if not os.path.exists(self.full_log_filename):
            open(self.full_log_filename, 'a').close()
        
        self._print("")
        self._print(f"{get_time()}")
        self._print(f"python {' '.join(sys.argv)}")
        self._print("")
    
    def log(self, line, print_in_file=True):
        """
        line viene stampata in stdout e self.full_log_filename.
        if print_in_file==True, viene stampata anche in self.log_filename
        """
        line = str(line)
        if self.print_day:
            line = get_time(with_date=True) + "   " + line
        elif self.print_time:
            line = get_time(with_date=False) + "   " + line
        self._print(line, print_in_file)
    
    def _print(self, line, print_in_file=True):
        print(line)
        if print_in_file:
            with open(self.log_filename, "a") as myfile:
                myfile.write(line + "\n")
        with open(self.full_log_filename, "a") as myfile:
            myfile.write(line + "\n")


def get_time(with_date=True):
    """Ritorna l'ora corrente hh:mm:ss"""
    import datetime
    if with_date:
        return str(datetime.datetime.now())[:19]
    else:
        return str(datetime.datetime.now())[11:19]

