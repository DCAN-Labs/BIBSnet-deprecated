#!/usr/bin/env python3
# coding: utf-8

"""
Wrapper to run nnU-Net_predict trained on BCP subjects
Greg Conan: gconan@umn.edu
Created: 2022-02-08
Updated: 2022-10-24
"""
# Import standard libraries
import argparse
from datetime import datetime 
from fnmatch import fnmatch
from glob import glob
import os
import pandas as pd
import subprocess
import sys


def main():
    # Time how long the script takes and get command-line arguments from user 
    start_time = datetime.now()
    cli_args = get_cli_args()

    run_nnUNet_predict(cli_args)

    # Show user how long the pipeline took and end the pipeline here
    exit_with_time_info(start_time)


def run_nnUNet_predict(cli_args):
    """
    Run nnU-Net_predict in a subshell using subprocess
    :param cli_args: Dictionary containing all command-line input arguments
    :return: N/A
    """
    subprocess.call((cli_args["nnUNet"], "-i",
                     cli_args["input"], "-o", cli_args["output"], "-t",
                     str(cli_args["task"]), "-m", cli_args["model"]))
    
    # Only raise an error if there are no output segmentation file(s)
    if not glob(os.path.join(cli_args["output"], "*.nii.gz")):
        # TODO This statement should change if we add a new model
        sys.exit("Error: Output segmentation file not created at the path "
                 "below during nnUNet_predict run.\n{}\n\nFor your input files "
                 "at the path below, check their filenames and visually "
                 "inspect them if needed.\n{}\n\n"
                 .format(cli_args["output"], cli_args["input"]))


def get_cli_args():
    """ 
    :return: Dictionary containing all validated command-line input arguments
    """
    script_dir = os.path.dirname(__file__)
    default_model = "3d_fullres"
    default_nnUNet_path = os.path.join(script_dir, "nnUNet_predict")
    default_task_ID = 512
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=valid_readable_dir, required=True,
        help=("Valid path to existing input directory with 1 T1w file and/or "
              "1 T2w file. T1w files should end with '_0000.nii.gz'. "
              "T2w files only should for T1w-and-T2w model(s). For T2w-only "
              "model(s), T2w files should end with '_0001.nii.gz'.")
    )
    parser.add_argument(
        "--output", "-o", type=valid_output_dir, required=True,
        help=("Valid path to a directory to save BIBSnet output files into. "
              "If this directory or its parent directory/ies does not exist, "
              "then they will automatically be created.")
    )
    parser.add_argument(
        "--nnUNet", "-n", type=valid_readable_file, default=default_nnUNet_path,
        help=("Valid path to existing executable file to run nnU-Net_predict. "
              "By default, this script will assume that nnU-Net_predict will "
              "be in the same directory as this script: {}".format(script_dir))
    )
    parser.add_argument(
        "--task", "-t", type=valid_whole_number, default=default_task_ID,
        help=("Task ID, which should be a 3-digit positive integer starting "
              "with 5 (e.g. 512). The default task ID is {}."
              .format(default_task_ID))
    )
    parser.add_argument(  # TODO Does this even need to be an argument, or will it always be the default?
        "--model", "-m", default=default_model,
        help=("Name of the nnUNet model to run. By default, it will run '{}'."
              .format(default_model))
    )
    return validate_cli_args(vars(parser.parse_args()), script_dir, parser)


def validate_cli_args(cli_args, script_dir, parser):
    """
    Verify that at least 1 T1w and/or 1 T2w file (depending on the task ID)
    exists in the --input directory
    :param cli_args: Dictionary containing all command-line input arguments
    :param script_dir: String, valid path to existing dir containing run.py
    :param parser: argparse.ArgumentParser to raise error if anything's invalid
    :return: cli_args, but with all input arguments validated
    """
    # Get info about which task ID(s) need T1s and which need T2s from .csv
    try:
        models_csv_path = os.path.join(script_dir, "models.csv")  # TODO Should we make this file path an input argument?
        tasks = pd.read_csv(models_csv_path, index_col=0)
        specified_task = tasks.loc[cli_args["task"]]

    # Verify that the specified --task number is a valid task ID
    except OSError:
        parser.error("{} not found. This file is needed to determine nnUNet "
                     "requirements for BIBSnet task {}."
                     .format(models_csv_path, cli_args["task"]))
    except KeyError:
        parser.error("BIBSnet task {0} is not in {1} so its requirements are "
                     "unknown. Add a task {0} row in that .csv or try one of "
                     "these tasks: {2}"
                     .format(cli_args["task"], models_csv_path, 
                             tasks.index.values.tolist()))

    # Validate that BIBSnet has all T1w/T2w input file(s) needed for --task
    err_msg = ("BIBSnet task {} requires image file(s) at the path(s) below, "
               "and at least 1 is missing. Either save the image file(s) "
               "there or try a different task.\n{}")
    img_glob_path = os.path.join(cli_args["input"], "*_000{}.nii.gz")
    how_many_T_expected = 0
    for t1or2 in (1, 2):
        # TODO Should this verify that ONLY one T1w file and/or ONLY one T2w file exists?
        if specified_task.get("T{}w".format(t1or2)):
            how_many_T_expected += 1
    img_files = glob(img_glob_path.format("?"))
    if how_many_T_expected == 2 and len(img_files) < 2:
        parser.error(err_msg.format(cli_args["task"], "\n".join((
            img_glob_path.format(0), img_glob_path.format(1)
        ))))
    elif how_many_T_expected == 1 and (
            len(img_files) < 1 or not fnmatch(img_files[0],
                                              img_glob_path.format(0))
        ):
        parser.error(err_msg.format(cli_args["task"], img_glob_path.format(0)))
        
    return cli_args


def valid_output_dir(path):
    """
    Try to make a folder for new files at path; throw exception if that fails
    :param path: String which is a valid (not necessarily real) folder path
    :return: String which is a validated absolute path to real writeable folder
    """
    return validate(path, lambda x: os.access(x, os.W_OK),
                    valid_readable_dir, "Cannot create directory at {}",
                    lambda y: os.makedirs(y, exist_ok=True))


def valid_readable_dir(path):
    """
    :param path: Parameter to check if it represents a valid directory path
    :return: String representing a valid directory path
    """
    return validate(path, os.path.isdir, valid_readable_file,
                    "Cannot read directory at '{}'")


def valid_readable_file(path):
    """
    Throw exception unless parameter is a valid readable filepath string. Use
    this, not argparse.FileType("r") which leaves an open file handle.
    :param path: Parameter to check if it represents a valid filepath
    :return: String representing a valid filepath
    """
    return validate(path, lambda x: os.access(x, os.R_OK),
                    os.path.abspath, "Cannot read file at '{}'")


def valid_whole_number(to_validate):
    """
    Throw argparse exception unless to_validate is a positive integer
    :param to_validate: Object to test whether it is a positive integer
    :return: to_validate if it is a positive integer
    """
    return validate(to_validate, lambda x: int(x) >= 0, int,
                    "{} is not a positive integer")


def validate(to_validate, is_real, make_valid, err_msg, prepare=None):
    """
    Parent/base function used by different type validation functions. Raises an
    argparse.ArgumentTypeError if the input object is somehow invalid.
    :param to_validate: String to check if it represents a valid object 
    :param is_real: Function which returns true iff to_validate is real
    :param make_valid: Function which returns a fully validated object
    :param err_msg: String to show to user to tell them what is invalid
    :param prepare: Function to run before validation
    :return: to_validate, but fully validated
    """
    try:
        if prepare:
            prepare(to_validate)
        assert is_real(to_validate)
        return make_valid(to_validate)
    except (OSError, TypeError, AssertionError, ValueError,
            argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError(err_msg.format(to_validate))


def exit_with_time_info(start_time, exit_code=0):
    """
    Terminate the pipeline after displaying a message showing how long it ran
    :param start_time: datetime.datetime object of when the script started
    :param exit_code: Int, exit code
    :return: N/A
    """
    print("BIBSnet for this subject took this long to run {}: {}"
          .format("successfully" if exit_code == 0 else "and then crashed",
                  datetime.now() - start_time))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
