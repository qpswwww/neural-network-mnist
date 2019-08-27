/************************************************************
  Copyright (C), 2018-2018
  FileName: mlpnet.cpp
  Author: Yongkang Zhang(School of Computer,Wuhan University)
  Version : 6.0
  Date: 2018/4/30
  Description: 多层感知机网络的结构体定义及其实现
***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <algorithm>
#include <time.h>
#include <cmath>

#define INPUT_MAXNUM 800 //输入层神经元个数上限
#define HIDDEN_MAXNUM 320 //隐含层神经元个数上限
#define OUTPUT_MAXNUM 10 //输出层神经元个数上限

#define SAMPLE_MAXNUM  130000 //训练样例数量
#define CHECK_MAXNUM 5000 //测试样例数量

using namespace std;

double rand_interval(double lowerBound,double upperBound) //在[lowerBound,upperBouond]之间随机一个数
{
    return (((double)(rand()%RAND_MAX))/((double)RAND_MAX))*(upperBound-lowerBound)+lowerBound;
}

double sigmoid(double x) //sigmoid函数,S(x)=1/(1+e^(-x))
{
    return 1/(1+exp(-x));
}

double d_sigmoid(double x) //sigmoid函数的导函数,S'(x)=S(x)(1-S(x))
{
    return sigmoid(x)*(1-sigmoid(x));
}

/*
结构体数据项：
    input_num,hidden_num,output_num:输入层、隐含层、输出层神经元个数
    net_error:当前神经网络输出与预期输出之间的均方差误差
    RATE_INPUT2HIDDEN,RATE_HIDDEN2OUTPUT,RATE_BIAS_HIDDEN,RATE_BIAS_OUTPUT:
    网络参数，分别为输入层到隐含层学习率,隐含层到输出层学习率,隐含层偏置学习率,输出层偏置学习率
    input[]:输入层神经元的值
    hidden_net[]:隐含层神经元Net值
    hidden_out[]:隐含层神经元Out值，Out=Sigmoid(Net)
    output_net[]:输出层神经元Net值
    output_out[]:输出层神经元Out值
    output_expectation[]:期望输出值
    sumg_w_input2hidden[][]:输入层神经元到隐含层神经元的每条边的权重梯度之和
    sumg_w_hidden2output[][]:隐含层神经元到输出层神经元的每条边的权重梯度之和
    w_input2hidden[][]:输入层神经元到隐含层神经元的边的权重
    w_hidden2output[][]:隐含层神经元到输出层神经元的边的权重
    bias_hidden:隐含层偏置值
    bias_output:输出层偏置值
    sumg_bias_hidden:隐含层偏置值梯度之和
    sumg_bias_output:输出层偏置值梯度之和
*/
struct NeuralNetwork
{
    int input_num,hidden_num,output_num; //输入层、隐含层、输出层神经元个数

    double net_error; //当前神经网络输出与预期输出之间的均方差误差

    double RATE_INPUT2HIDDEN=0.01; //输入层到隐含层学习率
    double RATE_HIDDEN2OUTPUT=0.01; //隐含层到输出层学习率
    double RATE_BIAS_HIDDEN=0.01; //隐含层偏置学习率
    double RATE_BIAS_OUTPUT=0.01; //输出层偏置学习率

    double input[INPUT_MAXNUM]; //输入层神经元的值
    
    double hidden_net[HIDDEN_MAXNUM]; //隐含层神经元Net值
    double hidden_out[HIDDEN_MAXNUM]; //隐含层神经元Out值，Out=Sigmoid(Net)
    
    double output_net[OUTPUT_MAXNUM]; //输出层神经元Net值
    double output_out[OUTPUT_MAXNUM]; //输出层神经元Out值

    double output_expectation[OUTPUT_MAXNUM]; //期望输出值
    
    double sumg_w_input2hidden[INPUT_MAXNUM][HIDDEN_MAXNUM]; //输入层神经元到隐含层神经元的每条边的权重梯度之和
    double sumg_w_hidden2output[HIDDEN_MAXNUM][OUTPUT_MAXNUM]; //隐含层神经元到输出层神经元的每条边的权重梯度之和

    double w_input2hidden[INPUT_MAXNUM][HIDDEN_MAXNUM]; //输入层神经元到隐含层神经元的边的权重
    double w_hidden2output[HIDDEN_MAXNUM][OUTPUT_MAXNUM]; //隐含层神经元到输出层神经元的边的权重

    double bias_hidden; //隐含层偏置值
    double bias_output; //输出层偏置值
    
    double sumg_bias_hidden; //隐含层偏置值梯度之和
    double sumg_bias_output; //输出层偏置值梯度之和

    /*
    函数功能：对神经网络权值(w_input2hidden[][],w_hidden2output[][],bias_hidden,bias_output)进行初始化
    入口参数：无
    出口参数：无
    函数返回值：无
    */
    void network_init() //神经网络权值初始化
    {
        net_error=1e18;

        for(int i=0;i<input_num;i++)
            for(int j=0;j<hidden_num;j++)
                w_input2hidden[i][j]=rand_interval(-0.01,0.01);
        
        for(int i=0;i<hidden_num;i++)
            for(int j=0;j<output_num;j++)
                w_hidden2output[i][j]=rand_interval(-0.01,0.01);

        bias_hidden=rand_interval(-0.01,0.01);
        bias_output=rand_interval(-0.01,0.01);
    }

    /*
    函数功能：保存已训练好的神经网络的参数
            输出文件格式:
                input_num,hidden_num,output_num
                net_error,RATE_INPUT2HIDDEN,RATE_HIDDEN2OUTPUT,RATE_BIAS_HIDDEN,RATE_BIAS_OUTPUT
                w_input2hidden
                w_hidden2output
                bias_hidden,bias_output
    入口参数：file_dir(输出文件目录及文件名)
    出口参数：无
    函数返回值：无
    */
    void save_net(string file_dir)
    {
        freopen(file_dir.c_str(),"w",stdout);

        printf("%d %d %d\n",input_num,hidden_num,output_num); //输入层、隐含层、输出层神经元个数
        printf("%lf %lf %lf %lf %lf\n",net_error,RATE_INPUT2HIDDEN,RATE_HIDDEN2OUTPUT,RATE_BIAS_HIDDEN,RATE_BIAS_OUTPUT);
        
        for(int i=0;i<input_num;i++)
        {
            for(int j=0;j<hidden_num;j++)
                printf("%lf ",w_input2hidden[i][j]);
            printf("\n");
        }
        
        for(int i=0;i<hidden_num;i++)
        {
            for(int j=0;j<output_num;j++)
                printf("%lf ",w_hidden2output[i][j]);
            printf("\n");
        }
        
        printf("%lf %lf\n",bias_hidden,bias_output);

        fclose(stdout);
    }

    /*
    函数功能：读入一个训练好的神经网络的各参数
            输入文件格式:
                input_num,hidden_num,output_num
                net_error,RATE_INPUT2HIDDEN,RATE_HIDDEN2OUTPUT,RATE_BIAS_HIDDEN,RATE_BIAS_OUTPUT
                w_input2hidden
                w_hidden2output
                bias_hidden,bias_output
    入口参数：file_dir(输入文件目录及文件名)
    出口参数：无
    函数返回值：无
    */
    void load_net(string file_dir)
    {
        freopen(file_dir.c_str(),"r",stdin);

        scanf("%d%d%d",&input_num,&hidden_num,&output_num); //输入层、隐含层、输出层神经元个数
        scanf("%lf%lf%lf%lf%lf",&net_error,&RATE_INPUT2HIDDEN,&RATE_HIDDEN2OUTPUT,&RATE_BIAS_HIDDEN,&RATE_BIAS_OUTPUT);
        
        for(int i=0;i<input_num;i++)
            for(int j=0;j<hidden_num;j++)
                scanf("%lf",&w_input2hidden[i][j]);
        
        for(int i=0;i<hidden_num;i++)
            for(int j=0;j<output_num;j++)
                scanf("%lf",&w_hidden2output[i][j]);
        
        scanf("%lf%lf",&bias_hidden,&bias_output);

        fclose(stdin);
    }

    /*
    函数功能：从输入数据正向传播得到神经网络的预测输出
    入口参数：input[]（输入数据）
    出口参数：output_out[](神经网络的预测输出)
    函数返回值：无
    */
    void forward_propagation()
    {
        //更新隐含层神经元的Net、Out值
        for(int i=0;i<hidden_num;i++)
        {
            hidden_net[i]=0;
            for(int j=0;j<input_num;j++)
                hidden_net[i]+=input[j]*w_input2hidden[j][i];
            hidden_net[i]+=bias_hidden;
            hidden_out[i]=sigmoid(hidden_net[i]);
        }

        //更新输出层神经元的Net、Out值
        for(int i=0;i<output_num;i++)
        {
            output_net[i]=0;
            for(int j=0;j<hidden_num;j++)
                output_net[i]+=hidden_out[j]*w_hidden2output[j][i];
            output_net[i]+=bias_output;
            output_out[i]=sigmoid(output_net[i]);
        }
    }

    /*
    函数功能：根据神经网络的预测输出和训练数据的真实输出之间的误差，
            反向传播得到神经网络各参数的梯度，并将每个参数的梯度累加到
            该参数在当前训练批次下的梯度之和 sumg 中。
    入口参数：output_out[](神经网络的预测输出),output_expectation[](训练数据的真实输出)
    出口参数：            
            sumg_w_input2hidden[][]:输入层神经元到隐含层神经元的每条边的权重梯度之和
            sumg_w_hidden2output[][]:隐含层神经元到输出层神经元的每条边的权重梯度之和
            sumg_bias_hidden:隐含层偏置值梯度之和
            sumg_bias_output:输出层偏置值梯度之和
    函数返回值：无
    */
    void backward_propagation() //神经网络误差的反向传播
    {
        for(int i=0;i<output_num;i++) //输出层神经元i
            for(int j=0;j<hidden_num;j++) //隐含层神经元j
            {
                double dE_div_dW=(output_expectation[i]-output_out[i])*output_out[i]*(1-output_out[i])*hidden_out[j];
                sumg_w_hidden2output[j][i]+=dE_div_dW;
            }

        double sigma_dE_div_d_hidden_out[HIDDEN_MAXNUM];

        for(int i=0;i<hidden_num;i++) //隐含层神经元i
        {
            sigma_dE_div_d_hidden_out[i]=0;
            
            for(int j=0;j<output_num;j++) //输出层神经元j
            {
                double dE_div_d_hidden_out=(output_expectation[j]-output_out[j])*output_out[j]*(1-output_out[j])*w_hidden2output[i][j];
                sigma_dE_div_d_hidden_out[i]+=dE_div_d_hidden_out;
            }

            for(int j=0;j<input_num;j++) //输入层神经元j
            {
                double d_hidden_out_div_dW=hidden_out[i]*(1-hidden_out[i])*input[j];
                double dE_div_dW=sigma_dE_div_d_hidden_out[i]*d_hidden_out_div_dW;
                sumg_w_input2hidden[j][i]+=dE_div_dW;
            }
        }

        for(int i=0;i<output_num;i++) //更新输出层偏置值
        {
            double dE_div_dbias=(output_expectation[i]-output_out[i])*output_out[i]*(1-output_out[i]);
            sumg_bias_output+=dE_div_dbias;
        }

        for(int i=0;i<hidden_num;i++) //更新隐含层偏置值
        {
            double d_hidden_out_div_dbias=hidden_out[i]*(1-hidden_out[i]);
            double dE_div_dbias=sigma_dE_div_d_hidden_out[i]*d_hidden_out_div_dbias;
            sumg_bias_hidden+=dE_div_dbias;
        }
    }

    /*
    函数功能：根据当前训练批次下每个参数的梯度总和 sumg，更新每个参数。
    入口参数：          
            sumg_w_input2hidden[][](输入层神经元到隐含层神经元的每条边的权重梯度之和)
            sumg_w_hidden2output[][](隐含层神经元到输出层神经元的每条边的权重梯度之和)
            sumg_bias_hidden(隐含层偏置值梯度之和
            sumg_bias_output(输出层偏置值梯度之和)
    出口参数： 
            w_input2hidden[][](输入层神经元到隐含层神经元的边的权重)
            w_hidden2output[][](隐含层神经元到输出层神经元的边的权重)
            bias_hidden(隐含层偏置值)
            bias_output(输出层偏置值)
    函数返回值：无
    */
    void update_weight()
    {
        for(int i=0;i<input_num;i++)
            for(int j=0;j<hidden_num;j++)
                w_input2hidden[i][j]+=RATE_INPUT2HIDDEN*sumg_w_input2hidden[i][j];
        
        for(int i=0;i<hidden_num;i++)
            for(int j=0;j<output_num;j++)
                w_hidden2output[i][j]+=RATE_HIDDEN2OUTPUT*sumg_w_hidden2output[i][j];
        
        bias_hidden+=RATE_BIAS_HIDDEN*sumg_bias_hidden;
        bias_output+=RATE_BIAS_OUTPUT*sumg_bias_output;
    }

    /*
    函数功能：输入一批验证数据，使用均方误差估计神经网络对于每个训练数据
            的预测输出与真实输出之间的误差，并输出所有验证数据的误差之和。
    入口参数：          
            check_num(验证数据个数)
            check_input[][](验证数据的输入)
            check_output[][](验证数据的真实输出)
    出口参数： 
            net_error(当前神经网络输出与预期输出之间的均方差误差)
    函数返回值：无
    */
    void get_error(int check_num,double check_input[CHECK_MAXNUM][INPUT_MAXNUM],double check_output[CHECK_MAXNUM][OUTPUT_MAXNUM]) //用测试样本来检验误差
    {
        net_error=0;
        for(int i=0;i<check_num;i++)
        {
            for(int j=0;j<input_num;j++)
                input[j]=check_input[i][j];

            forward_propagation();

            for(int j=0;j<output_num;j++)
                net_error+=(check_output[i][j]-output_out[j])*(check_output[i][j]-output_out[j]);
        }
        net_error/=check_num;
        net_error/=2;
    }

    /*
    函数功能：读入一批训练数据，使用这些训练数据更新一次整个网络的权值。
    入口参数：          
            sample_num(训练数据的个数)
            sample_input[][](训练数据的输入)
            sample_output[][](训练数据的真实输出)
    出口参数：无
    函数返回值：无
    */
    void train(int sample_num,double sample_input[SAMPLE_MAXNUM][INPUT_MAXNUM],double sample_output[SAMPLE_MAXNUM][OUTPUT_MAXNUM])
    {
        //梯度和的变量清零
        memset(sumg_w_input2hidden,0,sizeof(sumg_w_input2hidden));
        memset(sumg_w_hidden2output,0,sizeof(sumg_w_hidden2output));
        sumg_bias_hidden=0;
        sumg_bias_output=0;
        
        //将mini-batch中的样本全部放入神经网络里训练一次，求出各个权值上的梯度和
        for(int i=0;i<sample_num;i++)
        {
            for(int j=0;j<input_num;j++)
                input[j]=sample_input[i][j];
            for(int j=0;j<output_num;j++)
                output_expectation[j]=sample_output[i][j];
            
            forward_propagation(); //在神经网络中正向传播输入数据
            backward_propagation(); //在神经网络中反向传播误差以更新各权值
        }

        update_weight(); //更新权值
    }

    /*
    函数功能：读入一个测试数据的输入，输出神经网络的预测结果
    入口参数：          
            data_input[][](测试数据的输入)
    出口参数：
            data_output[][](测试数据的预测输出)
    函数返回值：无
    */
    void recognize(double data_input[INPUT_MAXNUM],double data_output[OUTPUT_MAXNUM]) //输入数据data_input,获得神经网络的输出data_output
    {
        double minv=100000,maxv=-100000;
        for(int i=0;i<input_num;i++)
        {
            minv=min(minv,data_input[i]);
            maxv=max(maxv,data_input[i]);
        }
        for(int i=0;i<input_num;i++)
            input[i]=(data_input[i]-minv)/(maxv-minv); //MinMax归一化输入数据

        /*for(int i=0;i<input_num;i++)
            input[i]=data_input[i]/255;*/
        forward_propagation();
        for(int i=0;i<output_num;i++)
            data_output[i]=output_out[i];
    }
};