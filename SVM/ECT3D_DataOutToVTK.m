function ECT3D_DataOutToVTK(filename,vtx,simp,vol,pic,SimpMap)
fid = fopen(filename,'w');
fprintf(fid,'# vtk DataFile Version 3.0\n');
fprintf(fid,'Reconstructed 3DECT image\n');
fprintf(fid,'ASCII\n');
fprintf(fid,'DATASET UNSTRUCTURED_GRID\n');
fprintf(fid,'POINTS %d float\n',size(vtx,1));

if (min(pic) <= 0)
    minimum=0;
else
    minimum = min(pic);
end

if (max(pic) > 1)
    maximum=max(pic);
else
    maximum = 1;
end

for i=1:size(vtx,1)
    fprintf(fid,'%6f %6f %6f\n',vtx(i,1),vtx(i,2),vtx(i,3));
end

% fprintf(fid,'0 0 0\n');
% fprintf(fid,'0.1 0.1 0.1\n');

fprintf(fid,'\nCELLS %d %d\n',size(simp,1),size(simp,1)*5);
for i=1:size(simp,1)
    fprintf(fid,'4 %d %d %d %d\n',simp(i,1)-1,simp(i,2)-1,simp(i,3)-1,simp(i,4)-1);
end

fprintf(fid,'\nCELL_TYPES %d\n',size(simp,1));
for i=1:size(simp,1)
    fprintf(fid,'10\n');
end

fprintf(fid,'\nPOINT_DATA %d\n',size(vtx,1));

pic_cols = size(pic,2);

TempVTX = ones(size(vtx,1),1+pic_cols);

for i=1:size(vtx,1)
    TempVTX(i,1) = TempVTX(i,1)*i;
end

AverageVTXPermitt = zeros(1,pic_cols);
vl = 0;

% modified to use SimpMap matrix

% ECT3D_Progressbar
for i=1:size(TempVTX,1)
    vl = 0;
    for pcols=1:pic_cols
        for m=1:size(SimpMap,2)
            if (SimpMap(i,m)~=0)
                AverageVTXPermitt(pcols)  = AverageVTXPermitt(pcols) + (pic(SimpMap(i,m),pcols)*vol(SimpMap(i,m)));
                vl =  vl + vol(SimpMap(i,m));
            end
        end
        TempVTX(i,1+pcols) = AverageVTXPermitt(pcols) / vl;
    end
    AverageVTXPermitt = zeros(1,pic_cols);
% %     ECT3D_Progressbar(i/size(TempVTX,1))
end


for pcols=1:pic_cols
    fprintf(fid,'\nSCALARS Rozklad_materialu_%d float\n',pcols*10);
    fprintf(fid,'LOOKUP_TABLE default\n');
    for i=1:size(TempVTX,1)
        fprintf(fid,'%6.8f\n',TempVTX(i,1+pcols));
    end
end

fclose(fid);

s1 = 'C:\MayaVi\mayavi.exe -d ';
s2 = filename;
s3 = ' -m BandedSurfaceMap';

comm = strcat(s1,s2,s3);
system(comm);
