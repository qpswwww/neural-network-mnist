%************************************************************
%  Copyright (C), 2018-2018
%  FileName: data_generate.m
%  Author: Yongkang Zhang(School of Computer,Wuhan University)
%  Version : 1.0
%  Date: 2018/4/30
%  Description: 从train.csv中读入训练数据，通过弹性变换将数据扩充一倍
%               并保存至train_ext.csv
%***********************************************************

clc;
clear;
rand('seed',sum(100*clock))

mat=csvread('train.csv');
outmat=csvread('train_ext.csv');
inimg=uint8(zeros(28,28));

intensity=14;

n_gauss=13; %高斯核大小
sigma_gauss=3; %高斯核标准差
gauss_kernal=fspecial('gaussian',n_gauss,sigma_gauss);

for id=1:42000
    output_line=zeros(1,785);
    output_line(1,1)=mat(id,1);
    
    xdir=rand(28)*2-1;
    ydir=rand(28)*2-1;

    xdir=imfilter(xdir,gauss_kernal,'replicate');
    ydir=imfilter(ydir,gauss_kernal,'replicate');

    xdir=xdir*intensity;
    ydir=ydir*intensity;

    for i=1:28
        for j=1:28
            inimg(i,j)=mat(id,(i-1)*28+j+1);
        end
    end

    %figure();

    %subplot(1,2,1);
    %imshow(inimg);

    inimg=double(inimg);
    outimg=uint8(zeros(28));

    for i=1:28
        for j=1:28
            minx=i+int32(floor(xdir(i,j)));
            maxx=i+int32(ceil(xdir(i,j)));
            miny=j+int32(floor(ydir(i,j)));
            maxy=j+int32(ceil(ydir(i,j)));

            if(minx<1||miny<1||maxx>28||maxy>28)
                outimg(i,j)=0;
                continue;
            end

            tmp=(inimg(minx,miny)+inimg(minx,maxy)+inimg(maxx,miny)+inimg(maxx,maxy))/4;
            outimg(i,j)=uint8(tmp);
        end
    end

    %subplot(1,2,2);
    %imshow(outimg);
    for i=1:28
        for j=1:28
            output_line(1,(i-1)*28+j+1)=outimg(i,j);
        end
    end
    
    outmat=[outmat;output_line];
end

rowrank=randperm(size(outmat,1));
outmat=outmat(rowrank,:);

csvwrite('train_ext2.csv',outmat);