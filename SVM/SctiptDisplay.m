load ReconstructionDataBaseNL
load 3D_ECT_Combined_50itNL_RF0_8.mat

preds_hard = load('PhantomsPredicted_hard.mat');
preds_hard = preds_hard.PhantomDataBase;

preds_soft = load('PhantomsPredicted_soft.mat');
preds_soft = preds_soft.PhantomDataBase;

gt = load('PhantomDataBase.mat');
gt = gt.PhantomDataBase;

ECT3D_DataOutToVTK('C:\tmp\test2.vtk',vtx,simp,vol,preds_soft(:,30),SimpMap);