+++
draft = false
date = 2020-02-02T22:22:22+05:30
title = "Setting Up Paperspace for Deep Learning"
slug = ""
tags = ["DeepLearning"]
categories = []
math = "true"
+++

## Step by Step Guide to Setup Paperspace Machine for Deep Learning

- Create an Account : [paperspace.com](https://www.paperspace.com/)
- Sign up with my [promo code](https://www.paperspace.com/&R=DKCCRUZ) for Paperspace to get a $10 credit!
- Add in credit card information (required, even if you have a promo code)
- Go to https://www.paperspace.com/console/machines and click `New Machine`.
- Choose a region near to you.
- Choose `Ubuntu 16.04` in Linux Templates.
- Create new P4000 machine with at least 50 GB storage and a public IP and turn off Auto Snapshot (just for saving). And create the machine.

```
NVIDIA P4000
- Architecture: Pascal
- CUDA Cores: 1792
- GPU Memory: 8 GB GDDR5
- TeraFLOPS: 5.3
```

- You will get an email regarding your password and ssh command.
- Enter the console on your browser, insert your password and if you want to change the current password :
<kbd>$ passwd</kbd>

## SSH to paperspace machine from your local machine

- Install `ssh-copy-id`. Here's how to install it (Mac):

```bash
$ brew install ssh-copy-id
```

- `cd` into `~/.ssh` directory
- if you don't have an `.ssh` directory in your home folder, create it (`mkdir ~/.ssh`)
- if you don't have an `id_rsa.pub` file in your `~/.ssh` folder, create it (`ssh-keygen`)
- replace IP address in syntax below with your own, and run command

```bash
$ ssh-copy-id -i ~/.ssh/id_rsa.pub paperspace@74.###.###.###
```

- Use the password from the paperspace email.
- Add Paperspace info to `config` file
- make sure you are in the right directory (`cd ~/.ssh`)
- if you don't have a `config` file, create one.  This example creates file using nano editor (`nano config`)
- add these contents to your config file (replace IP address here with your Paperspace IP address)

```text
Host paperspace
     HostName 72.###.###.###
     IdentityFile ~/.ssh/id_rsa
     # StrictHostKeyChecking no  
     User paperspace
     LocalForward 8888 localhost:8888
```

- here's the nano command for saving file :  <kbd> ctrl o </kbd>  ->   <kbd> Enter </kbd>  
- here's the nano command for exiting a file :  <kbd> ctrl x </kbd>

## SSH from local machine and setting up Anaconda

- Fire up the terminal and type in : <kbd>ssh paperspace</kbd>
- If you want to change your password, here are the commands:
<kbd>passwd</kbd>

This will ask for the current password and then enter your password.

Next step after logging in the paperspace machine.

- Create a directory saying `Downloads`.

```
$ cd Downloads
$ wget -c https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
$ chmod +x Anaconda3-2018.12-Linux-x86_64.sh
$ ./Anaconda3-2018.12-Linux-x86_64.sh
$ export PATH=~/anaconda3/bin:$PATH
```

- Check conda version : <kbd>conda --version</kbd>

## Install pytorch

Install pytorch

```
$ pip install numpy torchvision_nightly
$ pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html
```

Checking if cuda is working:
```
$ python
>>> import torch
>>> torch.cuda.is_available()
```

If the above command returns False. Then we need to update/add the nvidia driver.
```
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt-get update
$ sudo apt-get install nvidia-396 nvidia-modprobe
```

Then restart the instance and check <kbd>$ nvdia-smi </kbd> and run pytorch command again and check.

This should solve the problem. If you still face some problems, ping in comments.

And the setup is complete. In the next post, we will build a model in pytorch.

**Remember to shut the notebook down and STOP Instance!**

- Paperspace has an option where you can choose to automatically shut down your machine at: 1 hr, 8 hrs, 1 day, 1 week. I chose the 8 hrs option.
- Note that the auto shut down works only for the browser console. Paperspace will not auto shut down your server if you SSH to it.
