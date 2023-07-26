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
    

def run_preBIBSnet(j_args, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: j_args, but with preBIBSnet working directory names added
    """
    completion_msg = "The anatomical images have been {} for use in BIBSnet"
    preBIBSnet_paths = get_and_make_preBIBSnet_work_dirs(j_args)
    sub_ses = get_subj_ID_and_session(j_args)

    # If there are multiple T1ws/T2ws, then average them
    create_anatomical_averages(preBIBSnet_paths["avg"], logger)  # TODO make averaging optional with later BIBSnet model?

    # Crop T1w and T2w images
    cropped = dict()
    crop2full = dict()
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]):
        cropped[t] = preBIBSnet_paths[f"crop_T{t}w"]
        crop2full[t] = crop_image(preBIBSnet_paths["avg"][f"T{t}w_avg"],
                                  cropped[t], j_args, logger)
    logger.info(completion_msg.format("cropped"))

    # Resize T1w and T2w images if running a BIBSnet model using T1w and T2w
    # TODO Make ref_img an input parameter if someone wants a different reference image?
    # TODO Pipeline should verify that reference_img files exist before running
    reference_img = os.path.join(SCRIPT_DIR, "data", "MNI_templates",
                                 "INFANT_MNI_T{}_1mm.nii.gz") 
    id_mx = os.path.join(SCRIPT_DIR, "data", "identity_matrix.mat")
    # TODO Resolution is hardcoded; infer it or get it from the command-line
    resolution = "1"  
    if j_args["ID"]["has_T1w"] and j_args["ID"]["has_T2w"]:
        msg_xfm = "Arguments for {}ACPC image transformation:\n{}"

        # Non-ACPC
        regn_non_ACPC = register_preBIBSnet_imgs_non_ACPC(
            cropped, preBIBSnet_paths["resized"], reference_img, 
            id_mx, resolution, j_args, logger
        )
        if j_args["common"]["verbose"]:
            logger.info(msg_xfm.format("non-", regn_non_ACPC["vars"]))

        # ACPC
        regn_ACPC = register_preBIBSnet_imgs_ACPC(
            cropped, preBIBSnet_paths["resized"], regn_non_ACPC["vars"],
            crop2full, preBIBSnet_paths["avg"], j_args, logger
        )
        if j_args["common"]["verbose"]:
            logger.info(msg_xfm.format("", regn_ACPC["vars"]))

        transformed_images = apply_final_prebibsnet_xfms(
            regn_non_ACPC, regn_ACPC, preBIBSnet_paths["avg"], j_args, logger
        )
        logger.info(completion_msg.format("resized"))

    # If running a T1w-only or T2w-only BIBSnet model, skip registration/resizing
    else:
        # Define variables and paths needed for the final (only) xfm needed
        t1or2 = 1 if j_args["ID"]["has_T1w"] else 2
        outdir = os.path.join(preBIBSnet_paths["resized"], "xfms")
        os.makedirs(outdir, exist_ok=True)
        out_img = get_preBIBS_final_img_fpath_T(t1or2, outdir, j_args["ID"])
        crop2BIBS_mat = os.path.join(outdir,
                                     "crop2BIBS_T{}w_only.mat".format(t1or2))
        out_mat = os.path.join(outdir, "full_crop_T{}w_to_BIBS_template.mat"
                                       .format(t1or2))

        run_FSL_sh_script(  # Get xfm moving the T1 (or T2) into BIBS space
            j_args, logger, "flirt", "-in", cropped[t1or2],
            "-ref", reference_img.format(t1or2), "-applyisoxfm", resolution,
            "-init", id_mx, # TODO Should this be a matrix that does a transformation?
            "-omat", crop2BIBS_mat
        )

        # Invert crop2full to get full2crop
        # TODO Move this to right after making crop2full, then delete the 
        #      duplicated functionality in align_ACPC_1_image
        full2crop = os.path.join(
            os.path.dirname(preBIBSnet_paths["avg"][f"T{t}w_avg"]),
            f"full2crop_T{t}w_only.mat"
        )
        run_FSL_sh_script(j_args, logger, "convert_xfm", "-inverse",
                          crop2full[t], "-omat", full2crop) 

        # - Concatenate crop .mat to out_mat (in that order) and apply the
        #   concatenated .mat to the averaged image as the output
        # - Treat that concatenated output .mat as the output to pass
        #   along to postBIBSnet, and the image output to BIBSnet
        run_FSL_sh_script(  # Combine ACPC-alignment with robustFOV output
            j_args, logger, "convert_xfm", "-omat", out_mat,
            "-concat", full2crop, crop2BIBS_mat
        )
        run_FSL_sh_script(  # Apply concat xfm to crop and move into BIBS space
            j_args, logger, "applywarp", "--rel", "--interp=spline",
            "-i", preBIBSnet_paths["avg"][f"T{t}w_avg"],
            "-r", reference_img.format(t1or2),
            "--premat=" + out_mat, "-o", out_img
        )
        transformed_images = {f"T{t1or2}w": out_img,
                              f"T{t1or2}w_crop2BIBS_mat": out_mat}

    # TODO Copy this whole block to postBIBSnet, so it copies everything it needs first
    # Copy preBIBSnet outputs into BIBSnet input dir
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]): 
        # Copy image files
        out_nii_fpath = j_args["optimal_resized"][f"T{t}w"]
        os.makedirs(os.path.dirname(out_nii_fpath), exist_ok=True)
        if j_args["common"]["overwrite"]:  # TODO Should --overwrite delete old image file(s)?
            os.remove(out_nii_fpath)
        if not os.path.exists(out_nii_fpath): 
            shutil.copy2(transformed_images[f"T{t}w"], out_nii_fpath)

        # Copy .mat into postbibsnet dir with the same name regardless of which
        # is chosen, so postBIBSnet can use the correct/chosen .mat file
        concat_mat = transformed_images[f"T{t}w_crop2BIBS_mat"]
        out_mat_fpath = os.path.join(  # TODO Pass this in (or out) from the beginning so we don't have to build the path twice (once here and once in postBIBSnet)
            j_args["optional_out_dirs"]["postbibsnet"],
            *sub_ses, "preBIBSnet_" + os.path.basename(concat_mat)
        )
        if not os.path.exists(out_mat_fpath):
            shutil.copy2(concat_mat, out_mat_fpath)
            if j_args["common"]["verbose"]:
                logger.info(f"Copying {concat_mat} to {out_mat_fpath}")
    logger.info("PreBIBSnet has completed")
    return j_args


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

def run_postBIBSnet(j_args, logger):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param logger: logging.Logger object to show messages and raise warnings
    :return: j_args, unchanged
    """
    sub_ses = get_subj_ID_and_session(j_args)

    # Template selection values
    age_months = j_args["ID"]["age_months"]
    logger.info("Age of participant: {} months".format(age_months))

    # Get template closest to age
    tmpl_age = get_template_age_closest_to(
        age_months, os.path.join(SCRIPT_DIR, "data", "chirality_masks")
    )
    if j_args["common"]["verbose"]:
        logger.info("Closest template-age is {} months".format(tmpl_age))

    # For left/right registration, use T1 for T1-only and T2 for T2-only, but
    # for T1-and-T2 combined use T2 for <22 months otherwise T1 (img quality)
    if j_args["ID"]["has_T1w"] and j_args["ID"]["has_T2w"]:
        t1or2 = 2 if int(age_months) < 22 else 1  # NOTE 22 cutoff might change
    elif j_args["ID"]["has_T1w"]:
        t1or2 = 1
    else:  # if j_args["ID"]["has_T2w"]:
        t1or2 = 2

    # Run left/right registration script and chirality correction
    left_right_mask_nifti_fpath = run_left_right_registration(
        sub_ses, tmpl_age, t1or2, j_args, logger
    )
    logger.info("Left/right image registration completed")

    # Dilate the L/R mask and feed the dilated mask into chirality correction
    if j_args["common"]["verbose"]:
        logger.info("Now dilating left/right mask")
    dilated_LRmask_fpath = dilate_LR_mask(
        os.path.join(j_args["optional_out_dirs"]["postbibsnet"], *sub_ses),
        left_right_mask_nifti_fpath
    )
    logger.info("Finished dilating left/right segmentation mask")
    nifti_file_paths, chiral_out_dir, xfm_ref_img_dict = run_correct_chirality(dilated_LRmask_fpath,
                                                      j_args, logger)
    for t in only_Ts_needed_for_bibsnet_model(j_args["ID"]):
        nii_outfpath = reverse_regn_revert_to_native(
            nifti_file_paths, chiral_out_dir, xfm_ref_img_dict[t], t, j_args, logger
        )
        
        logger.info("The BIBSnet segmentation has had its chirality checked and "
                    "registered if needed. Now making aseg-derived mask.")

        # TODO Skip mask creation if outputs already exist and not j_args[common][overwrite]
        aseg_mask = make_asegderived_mask(j_args, chiral_out_dir, t, nii_outfpath)  # NOTE Mask must be in native T1 space too
        logger.info(f"A mask of the BIBSnet T{t} segmentation has been produced")

        # Make nibabies input dirs
        bibsnet_derivs_dir = os.path.join(j_args["optional_out_dirs"]["derivatives"], 
                                    "bibsnet")
        derivs_dir = os.path.join(bibsnet_derivs_dir, *sub_ses, "anat")
        os.makedirs(derivs_dir, exist_ok=True)
        copy_to_derivatives_dir(nii_outfpath, derivs_dir, sub_ses, t, "aseg_dseg")
        copy_to_derivatives_dir(aseg_mask, derivs_dir, sub_ses, t, "brain_mask")
        input_path = os.path.join(j_args["common"]["bids_dir"],
                                               *sub_ses, "anat",
                                               f"*T{t}w.nii.gz")
        reference_path = glob(input_path)[0]
        generate_sidecar_json(sub_ses, reference_path, derivs_dir, t, "aseg_dseg")
        generate_sidecar_json(sub_ses, reference_path, derivs_dir, t, "brain_mask")

    # Copy dataset_description.json into bibsnet_derivs_dir directory for use in nibabies
    new_data_desc_json = os.path.join(bibsnet_derivs_dir, "dataset_description.json")
    if j_args["common"]["overwrite"]:
        os.remove(new_data_desc_json)
    if not os.path.exists(new_data_desc_json):
        shutil.copy2(os.path.join(SCRIPT_DIR, "data",
                                  "dataset_description.json"), new_data_desc_json)
    if j_args["common"]["work_dir"] == os.path.join("/", "tmp", "cabinet"):
        shutil.rmtree(j_args["common"]["work_dir"])
        logger.info("Working Directory removed at {}."
                    "To keep the working directory in the future,"
                    "set a directory with the --work-dir flag.\n"
                    .format(j_args['common']['work_dir']))
    logger.info("PostBIBSnet has completed.")
    return j_args


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
