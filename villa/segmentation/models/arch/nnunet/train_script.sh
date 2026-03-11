#PYTHONPATH=./ nnUNetv2_plan_and_preprocess -d 505 -pl nnUNetPlannerResEncM --verify_dataset_integrity -c 3d_fullres

PYTHONPATH=./ nnUNetv2_train 501 3d_192_group32_gelu 0  --c -p nnUNetResEncUNetMPlans -tr nnUNetTrainerMedialSurfaceRecall_MuSGD
PYTHONPATH=./ nnUNetv2_train 501 3d_192_group32_gelu 2  --c -p nnUNetResEncUNetMPlans -tr nnUNetTrainerMedialSurfaceRecall_MuSGD
PYTHONPATH=./ nnUNetv2_train 501 3d_192_group32_gelu 3  --c -p nnUNetResEncUNetMPlans -tr nnUNetTrainerMedialSurfaceRecall_MuSGD
PYTHONPATH=./ nnUNetv2_train 501 3d_192_group32_gelu 4  --c -p nnUNetResEncUNetMPlans -tr nnUNetTrainerMedialSurfaceRecall_MuSGD
#PYTHONPATH=./ nnUNetv2_train 501 3d_96_group_4 1  -p nnUNetResEncUNetMPlans
#PYTHONPATH=./ nnUNetv2_train 501 3d_96_group_2 1  -p nnUNetResEncUNetMPlans




