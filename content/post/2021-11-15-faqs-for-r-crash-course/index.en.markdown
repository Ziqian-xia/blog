---
title: FAQs for R Crash Course
author: Ziqian Xia
date: '2021-11-20'
slug: faqs-for-r-crash-course
categories: []
tags:
  - R-Teaching
subtitle: ''
summary: ''
authors: []
lastmod: '2021-11-15T14:46:32+08:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

这里是一些刚开始使用R语言将遇到的常见的问题：

1.  我无法安装Package

    可能的原因：

    -   未能正确设置路径(用户名可能是中文，导致R无法读取)

        -   解决方案：

            使用.libPaths()检查路径，是否存在???或乱码的情况

        修改.libPaths()

        如：.libPaths()\[1\]\<-"C:/Program Files/R"

2.  我安装了Package，但是不能运行函数

    -   请检查是否使用了library命令进行调用。

    -   重新使用install.packages()命令进行安装。

3.  为什么我的R界面没有Console

    -   请检查有没有安装R-Base
