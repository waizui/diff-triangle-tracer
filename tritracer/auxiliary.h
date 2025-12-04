/*
 Copyright (c) 2025 Changhe Liu
 Licensed under the Apache License, Version 2.0.

 For inquiries contact lch01234@gmail.com
*/

#pragma once
__device__ const float near_n = 0.2;
__device__ const float far_n = 1000.0;

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;

__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f, 0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};
