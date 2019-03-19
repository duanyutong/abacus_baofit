# -*- coding: utf-8 -*-
"""

create joint simulation of 16+20=36 boxes from two simulations

Created on Mon Mar 18 16:06:06 2019

@author: givoltage
"""

import os
from glob import glob
from shutil import copyfile
from tqdm import tqdm


def copy_sim(tag, sim_name_prefix, phases, phase_start, tagout):
    cosmology = 0
    redshift = 0.5
    store_dir = '/home/dyt/store/'
    from_dir = os.path.join(store_dir, sim_name_prefix+tag)
    save_dir = os.path.join(store_dir, 'joint_1100box_planck'+tagout)
    for phase in phases:
        print('Copying phase {} / {}...'.format(int(phase)+1, len(phases)))
        temp = os.path.join(
            from_dir,
            '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase),
            'z{}'.format(redshift), '*.txt')
        paths = glob(temp)
        for src in tqdm(paths):
            head, filename = os.path.split(src)
            head, foldername = os.path.split(head)
            dst = os.path.join(
                save_dir,
                'joint_1100box_planck_00-'+str(phase_start + phase),
                foldername, filename)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            copyfile(src, dst)


if __name__ == "__main__":

    tagout = ''  # '-mmatter'
    sim_name_prefix = 'emulator_1100box_planck'
    tag = '-recon'  # 'norsd'  # '# 'matter'  # 'z0.5'
    phases = range(16)  # range(16)  # [0, 1] # list(range(16))
    phase_start = 0
    copy_sim(tag, sim_name_prefix, phases, phase_start, tagout)

    sim_name_prefix = 'AbacusCosmos_1100box_planck'
    tag = ''
    phases = range(20)
    phase_start = 16
    copy_sim(tag, sim_name_prefix, phases, phase_start, tagout)
