/************************************************************
  Copyright (C), 2018-2018
  FileName: net_train.cpp
  Author: Yongkang Zhang(School of Computer,Wuhan University)
  Version : 6.0
  Date: 2018/4/30
  Description: 从train_ext2.txt中读入训练数据训练一个神经网络，
                并将训练好的网络参数保存至net_data.txt
***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <algorithm>
#include <time.h>
#include <cmath>
#include "mlpnet.cpp"

#define INPUT_MAXNUM 800 //输入层神经元个数上限
#define HIDDEN_MAXNUM 320 //隐含层神经元个数上限
#define OUTPUT_MAXNUM 10 //输出层神经元个数上限

#define SAMPLE_MAXNUM  130000 //训练样例数量
#define CHECK_MAXNUM 5000 //测试样例数量

using namespace std;

NeuralNetwork net;

double sample_input[SAMPLE_MAXNUM][INPUT_MAXNUM];
double sample_output[SAMPLE_MAXNUM][OUTPUT_MAXNUM];

double check_input[CHECK_MAXNUM][INPUT_MAXNUM];

double check_output[CHECK_MAXNUM][OUTPUT_MAXNUM];

FILE *TRAIN_FILE,*TEST_FILE;

/*
    函数功能：读入训练数据,并将其划分为训练组和验证组
    入口参数：          
            file_dir(训练数据的目录及文件名)
            sample_num(训练数据组数)
            check_num(验证数据组数)
            dimension(每组数据的维度)
    出口参数：
            sample_input[][](训练数据的输入)
            sample_output[][](训练数据的真实输出)
            check_input[][](验证数据的输入)
            check_output[][](验证数据的真实输出)
*/
void read_train_data(string file_dir,int sample_num,int check_num,int dimension)
{
    freopen(file_dir.c_str(),"r",stdin);

    for(int i=0;i<sample_num;i++) //读入第i个训练样本
    {
        int digit; //digit=该训练样本对应的数字
        scanf("%d",&digit);
        sample_output[i][digit]=1;
        for(int j=0;j<dimension;j++)
        {
            scanf("%lf",&sample_input[i][j]);
            sample_input[i][j]/=255.0;
        }
    }

    for(int i=0;i<check_num;i++) //读入第i个测试样本
    {
        int digit; //digit=该训练样本对应的数字
        scanf("%d",&digit);
        check_output[i][digit]=1;
        double minv=100000,maxv=-100000;
        for(int j=0;j<dimension;j++)
        {
            scanf("%lf",&check_input[i][j]);
            minv=min(minv,check_input[i][j]);
            maxv=max(maxv,check_input[i][j]);
        }
        for(int j=0;j<dimension;j++)
            check_input[i][j]=(check_input[i][j]-minv)/(maxv-minv); //MinMax归一化输入数据
    }

    fclose(stdin);
}

int train_id[SAMPLE_MAXNUM]; //train_id[i]记录在当前一轮训练中第i个用于训练的样本编号
double train_input[SAMPLE_MAXNUM][INPUT_MAXNUM],train_output[SAMPLE_MAXNUM][OUTPUT_MAXNUM];

int main()
{
    srand(time(0)); //确定随机种子

    int sample_num=122000,check_num=4000,dimension=784;
    int batch_size=100; //每轮训练中使用的训练数据数目
    int check_gap=100; //训练过程中每隔check_gap次就检验一次神经网络误差
    int max_epoch=16; //最大迭代次数

    read_train_data("E:/code/cpp/bpneuralnetwork/train_ext2.txt",sample_num,check_num,dimension);

    net.input_num=dimension;
    net.hidden_num=300;
    net.output_num=10;

    net.network_init();

    for(int i=0;i<sample_num;i++)
        train_id[i]=i;

    for(int epoch=1;epoch<=max_epoch&&net.net_error>0.001;epoch++)
    {
        random_shuffle(train_id,train_id+sample_num);

        int total_batch=sample_num/batch_size;
        for(int batch_id=0;batch_id<total_batch;batch_id++)
        {
            int batch_left=batch_id*batch_size;
            for(int i=0;i<batch_size;i++)
            {
                for(int j=0;j<net.input_num;j++)
                    train_input[i][j]=sample_input[train_id[batch_left+i]][j];
                for(int j=0;j<net.output_num;j++)
                    train_output[i][j]=sample_output[train_id[batch_left+i]][j];
            }
            net.train(batch_size,train_input,train_output);
            printf("Iteration: %d\n",(epoch-1)*total_batch+batch_id);
        }

        net.get_error(check_num,check_input,check_output);
        printf("Epoch:%d / Test Error:%.6lf\n",epoch,net.net_error);
    }

    printf("Final Test Error:%.6lf\n",net.net_error);

    net.save_net("E:/code/cpp/bpneuralnetwork/net_data.txt");

    return 0;
}