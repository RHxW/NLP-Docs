# nltk packages下载问题

使用python3的nltk包的时候需要先下载内部的packages：
```
import nltk
nltk.download()
```
执行命令后会弹出一个窗口，正常情况下点击download按钮即可。
但是我在下载的时候出现了两个问题，记录一下（macos）

## SSL验证错误
* 报错信息：
  
  `[SSL:CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate( _ssl.c:1108) `

* 解决办法：
  
  在命令行运行：
  ```
  import nltk
  import ssl
  try:
      _create_unverified_https_context = ssl._create_unverified_context
  except AttributeError:
      pass
  else:
      ssl._create_default_https_context = _create_unverified_https_context
  nltk.download()

  ```
执行之后SSL的报错信息就消失了，就可以正常下载了，但是有的时候会遇到下面这个问题

## HTTP ERROR 403
一般来说稍等一会就好，如果一直有问题的话可以尝试挂梯子或者直接从github打包下载原repo(https://github.com/nltk/nltk_data)再解压到相应位置（弹出框的下载位置）
