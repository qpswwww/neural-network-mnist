/************************************************************
  Copyright (C), 2018-2018
  FileName: net_recogn.cpp
  Author: Yongkang Zhang(School of Computer,Wuhan University)
  Version : 6.0
  Date: 2018/4/30
  Description: 从test.txt中读入测试数据，从net_data.txt中读入一个训练好的神经网络，
                并将每个数据的预测输出保存至label.csv
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

int main()
{
    srand(time(0)); //确定随机种子

    net.load_net("E:/code/cpp/bpneuralnetwork/net_data.txt");

    freopen("E:/code/cpp/bpneuralnetwork/test.txt","r",stdin);
    freopen("E:/code/cpp/bpneuralnetwork/label.csv","w",stdout);

    int data_num=28000,dimension=784; //数据数量、每个数据的维度

    printf("ImageId,Label\n");

    double data_input[INPUT_MAXNUM],data_output[OUTPUT_MAXNUM];

    for(int data_id=1;data_id<=data_num;data_id++)
    {
        for(int i=0;i<dimension;i++)
            scanf("%lf",&data_input[i]);
        
        net.recognize(data_input,data_output);

        int ans=0;
        double p_ans=-1e9;

        for(int i=0;i<10;i++)
        {
            if(data_output[i]>p_ans)
            {
                p_ans=data_output[i];
                ans=i;
            }
        } 

        printf("%d,%d\n",data_id,ans);
    }

    return 0;
}