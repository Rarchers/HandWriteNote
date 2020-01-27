# HandWriteNote
一个可以手写的草稿纸————python端       

[java端](https://github.com/Rarchers/HandWriteNote_kt) 请看这里
# 关于本项目：
本项目是HandWriteNote的python端代码，主要负责模型的训练以及调用，用于识别用户手写的数学算式。


# 我该如何使用本项目：

1.下载本项目放到D:\Coding\Project_Python\TF3.5中
2.导入pycharm
3.下载所需第三方库
4.开始搞事情

### 以下是文件内容介绍

>文件夹
>>fonts： 夹存放了各种字体，你可以在[gen.py](https://github.com/Rarchers/HandWriteNote/blob/master/gen.py) 中的gen_id_card类中进行修改使用的字体<br>
>>Image： 在测试中使用，存放的是生成的图像文件，如今已经不再使用<br>
>>Input： 用于实际使用，[CreateInput.py](https://github.com/Rarchers/HandWriteNote/blob/master/CreateInput.py) 生成的图像文件会被存放在这里，用于模拟真实情况中用户图像的存放<br>
>>model&&model1： 在定长度的识别训练中存放模型，只是中间模型，无需使用<br>
>>model_none： 不定长度识别中存放模型，是最终模型，在[Predict_NoneLength.py](https://github.com/Rarchers/HandWriteNote/blob/master/Predict_NoneLength.py) 中进行调用<br>
>>None_Image： 不定长度调用中生成的图像<br>
>>Result：其中都是txt文件，将识别的结果存入txt文件，便于java进行调用计算<br>
>>模型归档： 你可以在这里找到所有版本的模型，未指定不定长度都是默认18位数字或字符

>Python文件<br>
>>[Bar.py](https://github.com/Rarchers/HandWriteNote/blob/master/Bar.py) 没有任何用处，只是好玩，显示一个加载动画（实际上也就只是个动画）<br>
>>[CreateInput.py](https://github.com/Rarchers/HandWriteNote/blob/master/CreateInput.py) 用于生成图像并存入Input文件夹，模拟用户上传文件<br>
>>[FT.py](https://github.com/Rarchers/HandWriteNote/blob/master/FT.py) 已经废弃，不再介绍<br>
>>[gen.py](https://github.com/Rarchers/HandWriteNote/blob/master/gen.py) 图片的生成类，在这里随机生成图片以及图片的文字<br>
>>[NoneLength.py](https://github.com/Rarchers/HandWriteNote/blob/master/NoneLength.py) 训练不定长度的模型<br>
>>[Predict.py](https://github.com/Rarchers/HandWriteNote/blob/master/Predict.py) 已经废弃，不再介绍<br>
>>[Predict2.py](https://github.com/Rarchers/HandWriteNote/blob/master/Predict2.py) 在这里对定长度（18）的图片进行识别<br>
>>[Predict_NoneLength.py](https://github.com/Rarchers/HandWriteNote/blob/master/Predict_NoneLength.py) 对于不定长度的图片进行识别<br>
>>[Train.py](https://github.com/Rarchers/HandWriteNote/blob/master/Train.py) 训练定长度（18）模型


# 使用本项目过程中可能遇到的问题


    Q：不定长度 数字+四则运算+括号 是个什么垃圾正确率
    A：模型是我用我笔记本跑的(穷人买不起服务器)，所以在 7 和 / 的识别上有一定几率出错，如果你追求更高的精度，可以自行训练
    
    Q:我不想使用你那一串愚蠢的路径
    A:那就自己在每个预测文件中修改一下路径就好了
    
    Q:你这个cv2.imwrite怎么没有写入文件
    A:我也不知道，怪我咯，欢迎pr
    
    Q:为什么我用你的某些模型会报错呢
    A:你试着把预测的类和gen.py中定义的图片大小从（32，512）改为（32.256）
    
    Q: &%%^&*&%$^%&**$%^&*&^%$%^&
    A: 请提交Issue


