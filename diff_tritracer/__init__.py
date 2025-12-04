#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the Apache License, Version 2.0.
#
# For inquiries contact jan.held@uliege.be
#
# Additional modifications by:
# Copyright (c) 2025 Changhe Liu 
# Licensed under the Apache License, Version 2.0.
#
# For inquiries contact lch01234@gmail.com
#

import sysconfig
import torch
import torch.nn as nn
from typing import NamedTuple

from . import _C


class TracerSettings(NamedTuple):
    bg: torch.Tensor
    scale_modifier: float
    sh_degree: int
    debug: bool
    rebuild_gas: int  # 0 rebuild bvh, 1 update, 2 nothing
    extra_params: torch.Tensor


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


class _Tracer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tracer,
        ray_o,
        ray_d,
        triangles_points,
        sigma,
        num_points_per_triangle,
        cumsum_of_points_per_triangle,
        number_of_points,
        shs,
        colors_precomp,
        opacities,
        scaling,
        density_factor,
        tracer_settings: TracerSettings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            ray_o,
            ray_d,
            triangles_points,
            sigma,
            num_points_per_triangle,
            cumsum_of_points_per_triangle,
            number_of_points,
            shs,
            tracer_settings.sh_degree,
            colors_precomp,
            opacities,
            scaling,
            density_factor,
            tracer_settings.bg,
            tracer_settings.extra_params,
            tracer_settings.rebuild_gas,
            tracer_settings.debug,
        )

        # Invoke C++/CUDA/OptiX tracer
        if tracer_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                res = tracer.trace_forward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            res = tracer.trace_forward(*args)

        (
            color,
            others,
            gpu_time,
            scaling,
            density_factor,
            max_blending,
        ) = res.data

        # Keep relevant tensors for backward
        ctx.tracer_settings = tracer_settings
        ctx.tracer = tracer
        ctx.number_of_points = number_of_points
        ctx.save_for_backward(
            ray_o,
            ray_d,
            triangles_points,
            sigma,
            num_points_per_triangle,
            cumsum_of_points_per_triangle,
            color,
            others,
            colors_precomp,
            opacities,  # orginal: radii,
            shs,
            scaling,
        )

        return color, gpu_time, scaling, density_factor, others, max_blending

    @staticmethod
    def backward(ctx, grad_out_color, _, __, ___, grad_out_others, _____):
        number_of_points = ctx.number_of_points
        tracer_settings = ctx.tracer_settings
        tracer = ctx.tracer

        (
            ray_o,
            ray_d,
            triangles_points,
            sigma,
            num_points_per_triangle,
            cumsum_of_points_per_triangle,
            color,
            others,
            colors_precomp,
            opacities,
            shs,
            scaling,
        ) = ctx.saved_tensors

        args = (
            ray_o,
            ray_d,
            tracer_settings.bg,
            triangles_points,
            sigma,
            num_points_per_triangle,
            cumsum_of_points_per_triangle,
            color,
            others,
            number_of_points,
            shs,
            tracer_settings.sh_degree,
            opacities,
            grad_out_color,
            grad_out_others,
            scaling,
            tracer_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if tracer_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                res = tracer.trace_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            res = tracer.trace_backward(*args)

        (
            grad_triangles,
            grad_sigma,
            grad_opacities,
            grad_sh,
        ) = res.data

        # print(torch.max(torch.abs(grad_triangles)), torch.min(torch.abs(grad_triangles)))

        # grad_triangles = grad_triangles.reshape(-1, 8, 3)
        grad_triangles = grad_triangles.flatten(0)

        grad_sigma = grad_sigma.view(-1, 1)

        grads = (
            None,  # tracer,
            None,  # ray_o,
            None,  # ray_d,
            grad_triangles,
            grad_sigma,
            None,  # num_points_per_triangle,
            None,  # cumsum_of_points_per_triangle,
            None,  # number_of_points,
            grad_sh,
            None,  # grad_colors_precomp
            grad_opacities,
            None,  # scaling,
            None,  # density_factor,
            None,  # tracer_settings: TracerSettings,
        )

        return grads


class Tracer(nn.Module):
    def __init__(self, forwad_ptx, backward_ptx) -> None:
        super().__init__()
        pkg_dir = sysconfig.get_path("purelib") + "/diff_tritracer"
        self.tracer = _C.OptixTracer(pkg_dir, forwad_ptx, backward_ptx)

    def forward(
        self,
        ray_o,
        ray_d,
        triangles_points,
        sigma,
        num_points_per_triangle,
        cumsum_of_points_per_triangle,
        number_of_points,
        opacities,
        scaling,
        density_factor,
        shs,
        colors_precomp,
        tracer_settings: TracerSettings,
    ):
        # Check if colors or SHs are provided
        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if shs is None:
            shs = torch.Tensor([]).cuda()
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).cuda()

        # Invoke the autograd function
        return _Tracer.apply(
            self.tracer,
            ray_o,
            ray_d,
            triangles_points,
            sigma,
            num_points_per_triangle,
            cumsum_of_points_per_triangle,
            number_of_points,
            shs,
            colors_precomp,
            opacities,
            scaling,
            density_factor,
            tracer_settings,
        )
