---
date: "2016-04-27T00:00:00Z"
external_link: ""
image:
  caption: Tongji
  focal_point: Smart
links:
- icon: twitter
  icon_pack: fab
  name: Follow
  url: https://twitter.com/ziqian_xia
summary: IoT-based environmental behaviour research, A proposal.
tags:
- Behavioral Research
title: "Waste Sorting Behaviour Research: A Proposal"
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""
---

# Waste Sorting & Recycling Behavior
---
夏子谦
---
## 1	引言
巨大数量的生活垃圾正处于处置的困境之中，而垃圾分类与回收是使得这些废弃物资源得以充分利用的必要手段(Rousta et al., 2020)。由于垃圾分类与回收行为往往需要用户端的配合（源头分类），提高居民垃圾分类与回收的行为是当前亟需关注的问题。

当前已有大量成熟的垃圾分类行为研究，这些研究的主题主要可以被分为以下几支：

## 2	分类行为或意愿的决定因素研究
### I.	总览
这类研究是典型的亲环境行为研究，往往根据一些理论框架，使用问卷的方法进行数据采集。通过建立一个关联性模型去确定影响特定亲环境行为的重要因素。

### II.	理论框架
	
就使用的理论框架而言，大部分学者使用计划行为理论（TPB）对影响行为的意识因素进行了确定。作为说明行为的最有影响力和最基本的心理学理论之一，TPB通过态度、主观规范、与感知行为控制影响了垃圾分类的意愿与行为(Chao and Lam, 2011; Shuangli et al., 2020; S. Wang et al., 2020a)。

由于TPB没有考虑个人道德与义务对于垃圾分类的促进作用，许多研究通过结合其他理论将TPB扩展应用于垃圾分类领域以分析废物分类行为。规范激活理论（NAM）逐渐被学者应用于识别垃圾分类意愿的研究中(Onwezen et al., 2013; Zhang et al.,2019)。
NAM理论强调了个人道德意识、事件后果预期对于态度影响路径的中介作用。因此，在大部分研究中，我们可以看到学者会使用NAM+TPB的理论框架去设计问卷并进行心理学因素的测度。另一方面，情绪ABC理论、理性行为理论(Theory of Reasoned Action)也从不同的角度分析了垃圾分类意识形成的机理。但是就理论的完整程度和适用性而言，这些理论被认为是底层的（TPB可以被看作是TRA的完整版，而情绪往往会变成一个潜变量加入到其他的理论模型中）(Gao et al., 2017; Yuriev et al., 2020a)。

理论框架是可以发展的，这往往需要多个样本的实证检验。当前也有很多学者注重于发展一个更好的解释亲环境行为的理论框架，但更多情况下，一个完整的框架所需要的数据是很难获得的，更多的研究仅仅是在原有框架基础上增加1~2个可能的潜变量(Morren and Grinstein, 2021; Yuriev et al., 2020b) 。

### III.	数据与模型
在数据的采集上，大部分的行为研究都需要测度被试的心理因素，而这一点往往是通过问卷来获得的。而在具体的行为方面，目前主流有以下行为测度方法：
-	使用问卷的代理变量近似估计行为 (例如使用问卷中报告的垃圾分类频率去估计真实的分类行为(S. Wang et al., 2020b)，这本质是自我报告的行为)
-	观察法(指派研究者去记录行为，这是客观的行为记录，但往往是小样本的。)
-	借助客观记录(日记记录、视频记录，更多出现在干预研究中(Young et al., 2017)。)
-	基于调查的仿真数据（系统动力学、Anylogic等等）
-	大规模的客观数据：IoT数据，大样本、自然实验条件(Tiefenbeck et al., 2019; Xia et al., 2021; Xia and Liu, 2021)。

就使用的建模方法，其中主流是使用结构方程模型、Logistic回归模型去识别垃圾分类意愿的影响因素(Zhang et al., 2019)、也有一部分研究使用了模拟仿真的方法与问卷结合的方法对垃圾分类行为进行建模分析(Chen and Gao, 2021; Meng et al., 2019)。由于问卷往往是使用Likert量表的多因素定量数据，回归或路径建模便成为了建模的主流选择。

### IV.	近期研究(2019~2021)
-	社会比较对于垃圾分类行为的影响(Knickmeyer, 2020)
-	知识、个人道德、参与度对于垃圾分类行为的影响(Wang et al., 2020)
-	社会特征、政策公众接受度对垃圾分类行为的影响(Setiawan, 2020)
-	社会交互对垃圾分类行为的影响(Luo et al., 2020)
-	个人态度、设备可接近性、政府激励对垃圾分类行为的影响(Zhang et al., 2019)
-	感知利益、感知困难对垃圾分类行为的影响(Cudjoe et al., 2020)
-	社会特征、自我满足对垃圾分类行为的影响(Wang et al., 2020)

## 3	分类行为干预研究
### I.	总览
这类研究是基于上一节影响因素研究的进一步深入。影响因素研究仅仅带来了相关关系，干预研究这可以得到强因果关系的结论。干预研究的本质就是实验，使用某种刺激，观察被试在刺激后的行为变化。

### II.	干预因素
以下是一些常用的干预因素：
-	激励（物质激励或非物质激励）：给与被试一定的行为奖励或使用市场法则调节行为，这是通过利己主义进行的干预。
-	信息：提高被试某方面的知识、进行教育活动。
-	Nudges：使用反馈（比如告诉你自己消耗了多少能源，提供历史数据，激活亲环境行为）
-	社会比较：使用社会规范的手段进行干预，包含了同辈之间的比较，提供社会规范信息
-	动机干预：进行承诺、设定目标等。

### III.	干预数据与方法
短期干预数据大部分来自网络，亚马逊的MTurk之类的众包网站。通过设计一个干预环境，让被试进行模拟的实验室干预(Krupa et al., 2014)，或直接招募被试，在实验室中进行干预(Berger and Wyss, 2021)。而长期干预数据来自田野实验，或利用准自然实验去测度一个干预是否是有效的(Favuzzi et al., 2020)。

这类研究的底层逻辑很简单，所使用的统计方法也比较成熟：ANOVA, T-test之类的Contrast都可以进行分析。在实验前可以进行Power分析与预注册。
### IV.	近期研究
-	物质激励与非物质激励对垃圾分类行为的影响(Li et al., 2020)
-	Nudge政策对垃圾分类行为的影响(Zhang and Wang, 2020)
-	视觉信息干预对垃圾分类行为的影响(Lin et al., 2016)

## 4	元研究
### I.	总览
元研究是研究的研究。元分析（meta-analysis )是对众多现有实证文献的再次统计，通过对相关文献中的统计指标利用相应的统计公式，进行再一次的统计分析，从而可以根据获得的统计显著性等来分析两个变量间真实的相关关系。元分析被认为是可信度最高的，对当前研究的一个系统性的总结(Sovacool et al., 2018)。

### II.	行为影响因素或干预的元研究
Psychological strategies to promote household recycling. A systematic review with meta-analysis of validated field interventions (Varotto and Spagnolli, 2017)

## 5	研究方向
-	利用大规模数据，在环境学范式上，我们可以研究分类行为提高可再生资源量带来的减排效应(Kang et al., 2021; Song et al., 2018)。
-	参考这三篇文章进行干预实验：大规模的客观数据意味着我们不会遭受自我报告数据带来的偏差，还可以进行长期的实时监控，这对于干预实验是很有利的(Tiefenbeck, 2017; Tiefenbeck et al., 2019, 2018)。
-	亲环境行为的退出机制与僭越）。
-	与微观数据进行匹配，研究其他因素对环境行为的挤出效应（现在研究促进亲环境行为的很多，对于抑制环境行为的非常少。）


## References
Berger, S., Wyss, A.M., 2021. Measuring pro-environmental behavior using the carbon emission task. J. Environ. Psychol. 75, 101613. https://doi.org/10.1016/j.jenvp.2021.101613

Chao, Y.-L.L., Lam, S.-P.P., 2011. Measuring responsible environmental behavior: Self-reported and other-reported measures and their differences in testing a behavioral model. Environ. Behav. 43, 53–71. https://doi.org/10.1177/0013916509350849

Chen, L., Gao, M., 2021. Novel information interaction rule for municipal household waste classification behavior based on an evolving scale-free network. Resour. Conserv. Recycl. 168, 105445. https://doi.org/10.1016/j.resconrec.2021.105445

Cudjoe, D., Yuan, Q., Han, M.S., 2020. An assessment of the influence of awareness of benefits and perceived difficulties on waste sorting intention in Beijing. J. Clean. Prod. 272. https://doi.org/10.1016/j.jclepro.2020.123084

Favuzzi, N., Trerotoli, P., Forte, M.G., Bartolomeo, N., Serio, G., Lagravinese, D., Vino, F., 2020. Evaluation of an alimentary education intervention on school canteen waste at a primary school in Bari, Italy. Int. J. Environ. Res. Public Health 17. https://doi.org/10.3390/ijerph17072558

Gao, L., Wang, S., Li, J., Li, H., 2017. Application of the extended theory of planned behavior to understand individual’s energy saving behavior in workplaces. Resour. Conserv. Recycl. 127, 107–113. https://doi.org/10.1016/j.resconrec.2017.08.030

Kang, P., Song, G., Xu, M., Miller, T.R., Wang, H., Zhang, H., Liu, G., Zhou, Y., Ren, J., Zhong, R., Duan, H., 2021. Low-carbon pathways for the booming express delivery sector in China. Nat. Commun. 12, 1–8. https://doi.org/10.1038/s41467-020-20738-4

Knickmeyer, D., 2020. Social factors influencing household waste separation: A literature review on good practices to improve the recycling performance of urban areas. J. Clean. Prod. 245, 118605. https://doi.org/10.1016/j.jclepro.2019.118605

Krupa, J.S., Rizzo, D.M., Eppstein, M.J., Brad Lanute, D., Gaalema, D.E., Lakkaraju, K., Warrender, C.E., 2014. Analysis of a consumer survey on plug-in hybrid electric vehicles. Transp. Res. Part A Policy Pract. 64, 14–31.

Li, C., Wang, Y., Li, Y., Huang, Y., Harder, M.K., 2020. The incentives may not be the incentive: A field experiment in recycling of residential food waste. Resour. Conserv. Recycl. https://doi.org/10.1016/j.resconrec.2020.105316

Lin, Z.Y., Wang, X., Li, C.J., Gordon, M.P.R., Harder, M.K., 2016. Visual prompts or volunteer models: An experiment in recycling. Sustain. 8, 1–16. https://doi.org/10.3390/su8050458

Luo, H., Zhao, L., Zhang, Z., 2020. The impacts of social interaction-based factors on household waste-related behaviors. Waste Manag. 118, 270–280. https://doi.org/10.1016/j.wasman.2020.08.046

Meng, X., Tan, X., Wang, Y., Wen, Z., Tao, Y., Qian, Y., 2019. Investigation on decision-making mechanism of residents’ household solid waste classification and recycling behaviors. Resour. Conserv. Recycl. 140, 224–234. https://doi.org/10.1016/j.resconrec.2018.09.021

Morren, M., Grinstein, A., 2021. The cross-cultural challenges of integrating personal norms into the Theory of Planned Behavior: A meta-analytic structural equation modeling (MASEM) approach. J. Environ. Psychol. 75, 101593. https://doi.org/10.1016/j.jenvp.2021.101593

Onwezen, M.C., Antonides, G., Bartels, J., 2013. The Norm Activation Model: An exploration of the functions of anticipated pride and guilt in pro-environmental behaviour. J. Econ. Psychol. 39, 141–153. https://doi.org/10.1016/j.joep.2013.07.005

Rousta, K., Zisen, L., Hellwig, C., 2020. Household waste sorting participation in developing countries—A meta-analysis. Recycling 5. https://doi.org/10.3390/recycling5010006

Setiawan, R.P., 2020. Factors determining the public receptivity regarding waste sorting: A case study in Surabaya city, Indonesia. Sustain. Environ. Res. 30, 1–8. https://doi.org/10.1186/s42834-019-0042-3

Shuangli, P., Guijun, Z., Qun, C., 2020. The psychological decision-making process model of giving up driving under parking constraints from the perspective of sustainable traffic. Sustain. 12, 1–19. https://doi.org/10.3390/su12177152

Song, G., Zhang, H., Duan, H., Xu, M., 2018. Packaging waste from food delivery in China’s mega cities. Resour. Conserv. Recycl. 130, 226–227. https://doi.org/10.1016/j.resconrec.2017.12.007

Sovacool, B.K., Axsen, J., Sorrell, S., 2018. Promoting novelty, rigor, and style in energy social science: Towards codes of practice for appropriate methods and research design. Energy Res. Soc. Sci. 45, 12–42. https://doi.org/10.1016/j.erss.2018.07.007

Tiefenbeck, V., 2017. Bring behaviour into the digital transformation. Nat. Energy 2, 1–3. https://doi.org/10.1038/nenergy.2017.85

Tiefenbeck, V., Goette, L., Degen, K., Tasic, V., Fleisch, E., Lalive, R., Staake, T., 2018. Overcoming salience bias: How real-time feedback fosters resource conservation. Manage. Sci. 64, 1458–1476. https://doi.org/10.1287/mnsc.2016.2646

Tiefenbeck, V., Wörner, A., Schöb, S., Fleisch, E., Staake, T., 2019. Real-time feedback promotes energy conservation in the absence of volunteer selection bias and monetary incentives. Nat. Energy 4, 35–41. https://doi.org/10.1038/s41560-018-0282-1

Varotto, A., Spagnolli, A., 2017. Psychological strategies to promote household recycling. A systematic review with meta-analysis of validated field interventions. J. Environ. Psychol. 51, 168–188. https://doi.org/10.1016/j.jenvp.2017.03.011

Wang, Q., Long, X., Li, L., Kong, L., Zhu, X., Liang, H., 2020. Engagement factors for waste sorting in China: The mediating effect of satisfaction. J. Clean. Prod. 267, 122046. https://doi.org/10.1016/j.jclepro.2020.122046

Wang, S., Wang, J., Yang, S., Li, J., Zhou, K., 2020a. From intention to behavior: Comprehending residents’ waste sorting intention and behavior formation process. Waste Manag. 113, 41–50.

Wang, S., Wang, J., Yang, S., Li, J., Zhou, K., 2020b. From intention to behavior: Comprehending residents’ waste sorting intention and behavior formation process. Waste Manag. 113, 41–50. https://doi.org/10.1016/j.wasman.2020.05.031

Wang, Y., Long, X., Li, L., Wang, Q., Ding, X., Cai, S., 2020. Extending theory of planned behavior in household waste sorting in China: the moderating effect of knowledge, personal involvement, and moral responsibility. Environ. Dev. Sustain. https://doi.org/10.1007/s10668-020-00913-9

Xia, Z., Liu, Y., 2021. Aiding Pro-Environment Behavior Measurements by Internet of Things. Curr. Res. Behav. Sci. 2, 100055. https://doi.org/10.1016/j.crbeha.2021.100055

Xia, Z., Zhang, S., Tian, X., Liu, Y., 2021. Understanding waste sorting behavior and key influencing factors through internet of things : Evidence from college student community. Resour. Conserv. Recycl. 174, 105775. https://doi.org/10.1016/j.resconrec.2021.105775

Young, W., Russell, S. V., Robinson, C.A., Barkemeyer, R., 2017. Can social media be a tool for reducing consumers’ food waste? A behaviour change experiment by a UK retailer. Resour. Conserv. Recycl. 117, 195–203. https://doi.org/10.1016/j.resconrec.2016.10.016

Yuriev, A., Dahmen, M., Paillé, P., Boiral, O., Guillaumie, L., 2020a. Pro-environmental behaviors through the lens of the theory of planned behavior: A scoping review. Resour. Conserv. Recycl. 155, 104660. https://doi.org/10.1016/j.resconrec.2019.104660

Yuriev, A., Dahmen, M., Paillé, P., Boiral, O., Guillaumie, L., 2020b. Pro-environmental behaviors through the lens of the theory of planned behavior: A scoping review. Resour. Conserv. Recycl. 155, 104660. https://doi.org/10.1016/j.resconrec.2019.104660

Zhang, B., Lai, K. hung, Wang, B., Wang, Z., 2019. From intention to action: How do personal attitudes, facilities accessibility, and government stimulus matter for household waste sorting? J. Environ. Manage. 233, 447–458. https://doi.org/10.1016/j.jenvman.2018.12.059

Zhang, Z., Wang, X., 2020. Nudging to promote household waste source separation: Mechanisms and spillover effects. Resour. Conserv. Recycl. 162, 105054. https://doi.org/10.1016/j.resconrec.2020.105054


