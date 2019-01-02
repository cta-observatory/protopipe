import numpy as np
import yaml
import argparse


def load_config(name):
    try:
        with open(name, 'r') as stream:
            cfg = yaml.load(stream)
    except FileNotFoundError as e:
        print(e)
        raise
    return cfg


class SignalHandler():
    """ handles ctrl+c signals; set up via
        `signal_handler = SignalHandler()
        `signal.signal(signal.SIGINT, signal_handler)`
        # or for two step interupt:
        `signal.signal(signal.SIGINT, signal_handler.stop_drawing)`
    """
    def __init__(self):
        self.stop = False
        self.draw = True

    def __call__(self, signal, frame):
        if self.stop:
            print('you pressed Ctrl+C again -- exiting NOW')
            exit(-1)
        print('you pressed Ctrl+C!')
        print('exiting after current event')
        self.stop = True

    def stop_drawing(self, signal, frame):
        if self.stop:
            print('you pressed Ctrl+C again -- exiting NOW')
            exit(-1)

        if self.draw:
            print('you pressed Ctrl+C!')
            print('turn off drawing')
            self.draw = False
        else:
            print('you pressed Ctrl+C!')
            print('exiting after current event')
            self.stop = True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_argparser():
    from os.path import expandvars
    parser = argparse.ArgumentParser(description='')

    # Add configuration file
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('-o', '--outfile', type=str, required=True)
    parser.add_argument('-m', '--max_events', type=int, default=None,
                        help="maximum number of events considered per file")
    parser.add_argument('-i', '--indir', type=str,
                        default=expandvars("$CTA_DATA/Prod3b/Paranal"))
    parser.add_argument('-f', '--infile_list', type=str, default="", nargs='*',
                        help="give a specific list of files to run on")
    parser.add_argument('--cam_ids', type=str,
                        default=["LSTCam", "NectarCam"], nargs='*',
                        help="give the specific list of camera types to run on")

    parser.add_argument('--wave_dir', type=str, default=None,
                        help="directory where to find mr_filter. "
                             "if not set look in $PATH")
    parser.add_argument('--wave_temp_dir', type=str, default='/dev/shm/',
                        help="directory where mr_filter to store the temporary fits"
                             " files")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--wave', dest="mode", action='store_const',
                            const="wave", default="tail",
                            help="if set, use wavelet cleaning -- default")
    mode_group.add_argument('--tail', dest="mode", action='store_const',
                            const="tail",
                            help="if set, use tail cleaning, otherwise wavelets")

    return parser


def prod3b_tel_ids(array, site="south"):
    if array in [None, ""]:
        return None

    if site.lower() in ["north", "la palma", "lapalma", "spain", "canaries"]:
        if array in 'subarray_LSTs':
            tel_ids = np.arange(1, 4 + 1)
        elif array in 'subarray_MSTs':
            tel_ids = np.arange(5, 19 + 1)
        elif array in 'full_array':
            tel_ids = np.arange(1, 19 + 1)
        else:
            raise ValueError("array {} not supported".format(array))
    else:
        raise ValueError("site '{}' not known -- try again".format(site))

    return tel_ids
