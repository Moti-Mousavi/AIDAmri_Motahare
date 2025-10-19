#!/usr/bin/env python

# Motahare 31.03.2025 (changed for registration different Stroke_masks)

import sys
import os
import nibabel as nii
import numpy as np
import shutil
import glob
import subprocess
import shlex
from pathlib import Path


def remove_ext(path):
    """
    Entfernt sicher '.nii.gz' oder '.nii' von einem Dateinamen.
    """
    name = Path(path).name
    if name.lower().endswith('.nii.gz'):
        return name[:-7]
    elif name.lower().endswith('.nii'):
        return name[:-4]
    else:
        return name


def regABA2DTI(inputVolume, stroke_masks, refStroke_mask,
               T2data, brain_template, brain_anno, splitAnno, splitAnno_rsfMRI,
               anno_rsfMRI, bsplineMatrix, outfile):
    # Base name ohne Extension
    base = remove_ext(inputVolume)

    # Affine registration: register T2data to inputVolume
    outputT2w = os.path.join(outfile, base + '_T2w.nii.gz')
    outputAff = os.path.join(outfile, base + '_transMatrixAff.txt')

    command = f"reg_aladin -ref {inputVolume} -flo {T2data} -res {outputT2w} -rigOnly -aff {outputAff}"
    command_args = shlex.split(command)
    try:
        result = subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Output of {command}:\n{result.stdout}")
    except Exception as e:
        print(f'Error while executing the command: {command_args} Error: {str(e)}')
        raise

    # Resample split Annotation
    outputAnnoSplit = os.path.join(outfile, base + '_AnnoSplit.nii.gz')
    command = f"reg_resample -ref {brain_anno} -flo {splitAnno} -trans {bsplineMatrix} -inter 0 -res {outputAnnoSplit}"
    command_args = shlex.split(command)
    try:
        result = subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Output of {command}:\n{result.stdout}")
    except Exception as e:
        print(f'Error while executing the command: {command_args} Error: {str(e)}')
        raise

    command = f"reg_resample -ref {inputVolume} -flo {outputAnnoSplit} -trans {outputAff} -inter 0 -res {outputAnnoSplit}"
    command_args = shlex.split(command)
    try:
        result = subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Output of {command}:\n{result.stdout}")
    except Exception as e:
        print(f'Error while executing the command: {command_args} Error: {str(e)}')
        raise

    # Resample parental (rsfMRI) Annotation
    outputAnnoSplit_par = os.path.join(outfile, base + '_AnnoSplit_parental.nii.gz')
    command = f"reg_resample -ref {brain_anno} -flo {splitAnno_rsfMRI} -trans {bsplineMatrix} -inter 0 -res {outputAnnoSplit_par}"
    command_args = shlex.split(command)
    try:
        result = subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Output of {command}:\n{result.stdout}")
    except Exception as e:
        print(f'Error while executing the command: {command_args} Error: {str(e)}')
        raise

    command = f"reg_resample -ref {inputVolume} -flo {outputAnnoSplit_par} -trans {outputAff} -inter 0 -res {outputAnnoSplit_par}"
    command_args = shlex.split(command)
    try:
        result = subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Output of {command}:\n{result.stdout}")
    except Exception as e:
        print(f'Error while executing the command: {command_args} Error: {str(e)}')
        raise

    # Resample parental Annotation (non-split)
    outputAnno_par = os.path.join(outfile, base + '_Anno_parental.nii.gz')
    command = f"reg_resample -ref {brain_anno} -flo {anno_rsfMRI} -trans {bsplineMatrix} -inter 0 -res {outputAnno_par}"
    command_args = shlex.split(command)
    try:
        result = subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Output of {command}:\n{result.stdout}")
    except Exception as e:
        print(f'Error while executing the command: {command_args} Error: {str(e)}')
        raise

    command = f"reg_resample -ref {inputVolume} -flo {outputAnno_par} -trans {outputAff} -inter 0 -res {outputAnno_par}"
    command_args = shlex.split(command)
    try:
        result = subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Output of {command}:\n{result.stdout}")
    except Exception as e:
        print(f'Error while executing the command: {command_args} Error: {str(e)}')
        raise

    # Resample Template
    outputTemplate = os.path.join(outfile, base + '_Template.nii.gz')
    command = f"reg_resample -ref {inputVolume} -flo {brain_template} -cpp {outputAff} -res {outputTemplate}"
    command_args = shlex.split(command)
    try:
        result = subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Output of {command}:\n{result.stdout}")
    except Exception as e:
        print(f'Error while executing the command: {command_args} Error: {str(e)}')
        raise

    # Create DSI Studio folder
    outfileDSI = os.path.join(os.path.dirname(inputVolume), 'DSI_studio')
    if os.path.exists(outfileDSI):
        shutil.rmtree(outfileDSI)
    os.makedirs(outfileDSI)

    # Handle reference stroke mask if provided (optional)
    outputRefStrokeMaskAff = None
    if refStroke_mask is not None and os.path.exists(refStroke_mask):
        refMatrix = find_RefAff(inputVolume)[0]
        refMTemplate = find_RefTemplate(inputVolume)[0]
        outputRefStrokeMaskAff = os.path.join(outfile, base + '_refStrokeMaskAff.nii.gz')
        command = f"reg_resample -ref {refMTemplate} -flo {refStroke_mask} -cpp {refMatrix} -res {outputRefStrokeMaskAff}"
        command_args = shlex.split(command)
        try:
            result = subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"Output of {command}:\n{result.stdout}")
        except Exception as e:
            print(f'Error while executing the command: {command_args} Error: {str(e)}')
            raise
   
    # Process Stroke Masks (files ending with 'Stroke_mask.nii.gz')
    """    
        for mask in stroke_masks:
            if os.path.exists(mask):
                orig_base = Path(mask).name
                mask_base = remove_ext(mask)
                outputMask = os.path.join(outfile, mask_base + '.nii')
                #command = f"reg_resample -ref {inputVolume} -flo {mask} -inter 0 -cpp {outputAff} -res {outputMask}"
         
                command = f"reg_resample -ref {inputVolume} -flo {mask} -inter 0 -trans {outputAff} -res {outputMask}"
                command_args = shlex.split(command)
                try:
                    result = subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    print(f"Output of {command}:\n{result.stdout}")
                except Exception as e:
                    print(f'Error while executing the command: {command_args} Error: {str(e)}')
                    raise
                registered_stroke_masks.append((orig_base, outputMask))

    # For each registered stroke mask, perform superposition with parental annotation and generate a scaled version for DSI Studio
    for orig_base, reg_mask in registered_stroke_masks:
        dataAnno = nii.load(outputAnnoSplit_par)
        dataMask = nii.load(reg_mask)
        imgAnno = dataAnno.get_fdata()
        imgMask = dataMask.get_fdata()
        # Create binary mask
        imgMask[imgMask > 0] = 1
        imgMask[imgMask == 0] = 0
        superPosAnnoMask = imgMask * imgAnno
        outAnnoMask = os.path.join(outfile, remove_ext(orig_base) + '_Anno.nii')
        nii.save(nii.Nifti1Image(superPosAnnoMask, dataAnno.affine), outAnnoMask)
    
        # Create scaled version for DSI Studio
        mask_base = remove_ext(orig_base)
        outputMaskScaled = os.path.join(outfileDSI, mask_base + '_scaled.nii')
        superPosFlipped = np.flip(superPosAnnoMask, 2)
        scale = np.eye(4) * 10
        scale[3][3] = 1
        unscaledNiiDataMask = nii.Nifti1Image(superPosFlipped, dataMask.affine * scale)
        nii.save(unscaledNiiDataMask, outputMaskScaled)
        # Copy corresponding annotation text file
        src_file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)), 'lib', 'ARA_annotationR+2000.nii.txt')
        dst_file = os.path.join(outfileDSI, mask_base + '_scaled.txt')
        shutil.copyfile(src_file, dst_file)
    """
    registered_stroke_masks = []
    if stroke_masks is not None and len(stroke_masks) > 0:
        # Process Stroke Masks (files ending with 'Stroke_mask.nii.gz')   
        for mask in stroke_masks:
            if os.path.exists(mask):
                orig_base = Path(mask).name
                mask_base = remove_ext(mask)
                outputMask = os.path.join(outfile, mask_base + '.nii')
                #command = f"reg_resample -ref {inputVolume} -flo {mask} -inter 0 -cpp {outputAff} -res {outputMask}"
         
                command = f"reg_resample -ref {inputVolume} -flo {mask} -inter 0 -trans {outputAff} -res {outputMask}"
                command_args = shlex.split(command)
                try:
                    result = subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    print(f"Output of {command}:\n{result.stdout}")
                except Exception as e:
                    print(f'Error while executing the command: {command_args} Error: {str(e)}')
                    raise
                registered_stroke_masks.append((orig_base, outputMask))

    # For each registered stroke mask, perform superposition with parental annotation and generate a scaled version for DSI Studio
    for orig_base, reg_mask in registered_stroke_masks:
        dataAnno = nii.load(outputAnnoSplit_par)
        dataMask = nii.load(reg_mask)
        imgAnno = dataAnno.get_fdata()
        imgMask = dataMask.get_fdata()
        # Create binary mask
        imgMask[imgMask > 0] = 1
        imgMask[imgMask == 0] = 0
        superPosAnnoMask = imgMask * imgAnno
        outAnnoMask = os.path.join(outfile, remove_ext(orig_base) + '_Anno.nii')
        nii.save(nii.Nifti1Image(superPosAnnoMask, dataAnno.affine), outAnnoMask)
    
        # Create scaled version for DSI Studio
        mask_base = remove_ext(orig_base)
        outputMaskScaled = os.path.join(outfileDSI, mask_base + '_scaled.nii')
        superPosFlipped = np.flip(superPosAnnoMask, 2)
        scale = np.eye(4) * 10
        scale[3][3] = 1
        unscaledNiiDataMask = nii.Nifti1Image(superPosFlipped, dataMask.affine * scale)
        nii.save(unscaledNiiDataMask, outputMaskScaled)
        # Copy corresponding annotation text file
        src_file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)), 'lib', 'ARA_annotationR+2000.nii.txt')
        dst_file = os.path.join(outfileDSI, mask_base + '_scaled.txt')
        shutil.copyfile(src_file, dst_file)

        
    # Process the general brain mask
    mask_base = base
    outputMaskScaled = os.path.join(outfileDSI, mask_base + 'Mask_scaled.nii')
    dataMask = nii.load(os.path.join(outfile, base + '_mask.nii.gz'))
    imgMask = dataMask.get_fdata()
    imgMask = np.flip(imgMask, 2)
    scale = np.eye(4) * 10
    scale[3][3] = 1
    unscaledNiiDataMask = nii.Nifti1Image(imgMask, dataMask.affine * scale)
    nii.save(unscaledNiiDataMask, outputMaskScaled)

    # Process Allen Brain and scaled annotations
    outputAnnoScaled = os.path.join(outfileDSI, base + 'Anno_scaled.nii')
    outputAnnorparScaled = os.path.join(outfileDSI, base + 'AnnoSplit_parental_scaled.nii')
    outputAllenBScaled = os.path.join(outfileDSI, base + 'Allen_scaled.nii')
    src_file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)), 'lib', 'ARA_annotationR+2000.nii.txt')
    dst_file = os.path.join(outfileDSI, base + 'Anno_scaled.txt')
    shutil.copyfile(src_file, dst_file)
    src_file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)), 'lib', 'annoVolume+2000_rsfMRI.nii.txt')
    dst_file = os.path.join(outfileDSI, base + 'AnnoSplit_parental_scaled.txt')
    shutil.copyfile(src_file, dst_file)

    dataAnno = nii.load(os.path.join(outfile, base + '_AnnoSplit.nii.gz'))
    dataAnnorspar = nii.load(os.path.join(outfile, base + '_AnnoSplit_parental.nii.gz'))
    dataAllen = nii.load(os.path.join(outfile, base + '_Template.nii.gz'))
    imgTempAnno = dataAnno.get_fdata()
    imgTempAnnorspar = dataAnnorspar.get_fdata()
    imgTempAllen = dataAllen.get_fdata()
    imgTempAllen = np.flip(imgTempAllen, 2)
    imgTempAnno = np.flip(imgTempAnno, 2)
    imgTempAnnorspar = np.flip(imgTempAnnorspar, 2)
    scale = np.eye(4) * 10
    scale[3][3] = 1
    unscaledNiiDataAnno = nii.Nifti1Image(imgTempAnno, dataAnno.affine * scale)
    unscaledNiiDataAnnorspar = nii.Nifti1Image(imgTempAnnorspar, dataAnnorspar.affine * scale)
    unscaledNiiDataAllen = nii.Nifti1Image(imgTempAllen, dataAllen.affine * scale)
    nii.save(unscaledNiiDataAnno, outputAnnoScaled)
    nii.save(unscaledNiiDataAnnorspar, outputAnnorparScaled)
    nii.save(unscaledNiiDataAllen, outputAllenBScaled)

    if outputRefStrokeMaskAff is not None:
        os.remove(outputRefStrokeMaskAff)

    return outputAnnoSplit


def find_RefStroke(refStrokePath, inputVolume):
    path = glob.glob(os.path.join(refStrokePath, os.path.basename(inputVolume)[:9], '*', "anat", "*", "IncidenceData_mask.nii.gz"), recursive=False)
    return path


def find_RefAff(inputVolume):
    parent_dir = os.path.dirname(os.path.dirname(inputVolume))
    path = glob.glob(os.path.join(parent_dir, 'anat', '*MatrixAff.txt'))
    return path


def find_RefTemplate(inputVolume):
    parent_dir = os.path.dirname(os.path.dirname(inputVolume))
    path = glob.glob(os.path.join(parent_dir, 'anat', '*TemplateAff.nii.gz'))
    return path


def find_relatedData(pathBase):
    # Find T2 data
    pathT2 = glob.glob(os.path.join(pathBase, 'anat', '*Bet.nii.gz'), recursive=False)

    # Original stroke masks
    stroke_masks = glob.glob(os.path.join(pathBase, 'anat', '*Stroke_mask.nii.gz'), recursive=False)

    # Modified by Motahare: replace items of stroke_masks list
    old_str, new_str = ('_ses-P3_', '_ses-P1_')
    for index, stroke_mask in enumerate(stroke_masks):
        basename = os.path.basename(stroke_mask)
        if old_str in basename:
            # Swap in the P1 version if it exists
            p1_mask = os.path.join(os.path.dirname(stroke_mask), basename.replace(old_str, new_str))
            if os.path.isfile(p1_mask):
                stroke_masks[index] = p1_mask

    # Now find the rest
    pathAnno      = glob.glob(os.path.join(pathBase, 'anat', '*Anno.nii.gz'),       recursive=False)
    pathAllen     = glob.glob(os.path.join(pathBase, 'anat', '*Allen.nii.gz'),      recursive=False)
    bsplineMatrix = glob.glob(os.path.join(pathBase, 'anat', '*MatrixBspline.nii'), recursive=False)

    # Return the adjusted stroke_masks, not the old list
    return pathT2, stroke_masks, pathAnno, pathAllen, bsplineMatrix



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Registration Allen Brain to DTI')
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--inputVolume', help='Path to the BET file of DTI data after preprocessing', required=True)
    parser.add_argument('-r', '--referenceDay', help='Reference Stroke mask (for example: P5)', nargs='?', type=str, default=None)
    parser.add_argument('-s', '--splitAnno', help='Split annotations atlas', nargs='?', type=str, 
                        default=os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)) + '/lib/ARA_annotationR+2000.nii.gz')
    parser.add_argument('-f', '--splitAnno_rsfMRI', help='Split annotations atlas for rsfMRI/DTI', nargs='?', type=str, 
                        default=os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)) + '/lib/annoVolume+2000_rsfMRI.nii.gz')
    parser.add_argument('-a', '--anno_rsfMRI', help='Parental Annotations atlas for rsfMRI/DTI', nargs='?', type=str, 
                        default=os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)) + '/lib/annoVolume.nii.gz')
    args = parser.parse_args()

    inputVolume = args.inputVolume
    if not os.path.exists(inputVolume):
        sys.exit(f"Error: '{inputVolume}' does not exist.")

    outfile = os.path.join(os.path.dirname(inputVolume))
    if not os.path.exists(outfile):
        os.makedirs(outfile)

    # Find related data
    base_dir = os.path.dirname(outfile)
    #pathT2, pathStroke_mask, pathAnno, pathTemplate, bsplineMatrix = find_relatedData(base_dir)
    pathT2, stroke_masks, pathAnno, pathTemplate, bsplineMatrix = find_relatedData(base_dir)
    print("→ final stroke_masks list:", stroke_masks)



    if len(pathT2) == 0:
        sys.exit(f"Error: {os.path.basename(inputVolume)} has no reference T2 template.")
    else:
        T2data = pathT2[0]

    #if len(pathStroke_mask) == 0:
    #   print(f"Notice: '{os.path.basename(inputVolume)}' has no defined stroke masks of type '*Stroke_mask.nii.gz' - will proceed without them.")
    #  stroke_masks = []
    #else:
    #   stroke_masks = pathStroke_mask

    if len(pathAnno) == 0:
        sys.exit(f"Error: {os.path.basename(inputVolume)} has no reference annotations.")
    else:
        brain_anno = pathAnno[0]

    if len(pathTemplate) == 0:
        sys.exit(f"Error: {os.path.basename(inputVolume)} has no reference template.")
    else:
        brain_template = pathTemplate[0]

    if len(bsplineMatrix) == 0:
        sys.exit(f"Error: {os.path.basename(inputVolume)} has no bspline Matrix.")
    else:
        bsplineMatrix = bsplineMatrix[0]

    # Handle reference stroke mask if provided
    refStroke_mask = None
    if args.referenceDay is not None:
        referenceDay = args.referenceDay
        refStrokePath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(outfile))), referenceDay)
        if not os.path.exists(refStrokePath):
            sys.exit(f"Error: '{refStrokePath}' does not exist.")
        refMasks = find_RefStroke(refStrokePath, inputVolume)
        if len(refMasks) == 0:
            print(f"Notice: '{os.path.basename(inputVolume)}' has no defined reference stroke mask - will proceed without it.")
        else:
            refStroke_mask = refMasks[0]

    if args.splitAnno is not None:
        splitAnno = args.splitAnno
    if not os.path.exists(splitAnno):
        sys.exit(f"Error: '{splitAnno}' does not exist.")

    if args.splitAnno_rsfMRI is not None:
        splitAnno_rsfMRI = args.splitAnno_rsfMRI
    if not os.path.exists(splitAnno_rsfMRI):
        sys.exit(f"Error: '{splitAnno_rsfMRI}' does not exist.")

    if args.anno_rsfMRI is not None:
        anno_rsfMRI = args.anno_rsfMRI
    if not os.path.exists(anno_rsfMRI):
        sys.exit(f"Error: '{anno_rsfMRI}' does not exist.")

    output = regABA2DTI(inputVolume, stroke_masks, refStroke_mask,
                        T2data, brain_template, brain_anno, splitAnno, splitAnno_rsfMRI,
                        anno_rsfMRI, bsplineMatrix, outfile)

    current_dir = os.path.dirname(inputVolume)
    search_string = os.path.join(current_dir, "*dwi.nii.gz")
    currentFile = glob.glob(search_string)

    search_string = os.path.join(current_dir, "*.nii*")
    created_imgs = glob.glob(search_string, recursive=True)

    os.chdir(os.path.dirname(os.getcwd()))
    for idx, img in enumerate(created_imgs):
        if img is None:
            continue
        # Optionally, you can adjust orientation using an external script.
        # os.system('python adjust_orientation.py -i ' + str(img) + ' -t ' + currentFile[0])
        continue

    print("Registration completed")
