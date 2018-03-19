# Linear Regression
## Gradient Decent
### Gradient
* Scalar?  \partial L / \partial w
function o f input
![](Linear%20Regression/D1F67256-903D-4C75-8248-451FEA9EB749.png)
* Vector? \partial L / \partial **w**
![](Linear%20Regression/33409916-FB83-40F1-B8BA-0AF61AEE51D4.png)
![](Linear%20Regression/5F9EA6EF-569A-46F8-8620-B3265C5C0DD9.png)
### Convex Function
only one place has a zero gradient: the optimal solution.
1. Compute the gradient of Loss function
2. Set gradient = 0
## Least Square Regression
![](Linear%20Regression/546FF3E8-4606-4697-B83A-EAEA6E320F71.png)
### A problem
* Assumption: training data should look like linear
* Assumption: loss function should look like convex
* Assumption: invertible
### Likelihood vs. Prior
Likelihood: P( data | parameter) — what you observed from data
Prior: P(parameter) — what you believe in mind
#### Posterior: using Bayes’ Rule
P( w=0.5 | data) = p(data | w =0.5) p(w=0.5) / p(data)
![](Linear%20Regression/9546275F-9DD3-4646-ADA6-5F8DD9CF8889.png)
![](Linear%20Regression/0DBA9B10-4CC2-456A-A92D-C913C5EE4ECC.png)
#### Occam’s Razor
prefer simplest solution
## Ridge Regression
![](Linear%20Regression/B42824FA-1FB6-4935-8263-26EA91C3A554.png)
## Polynomial Curve Fitting
![](Linear%20Regression/2C15FE69-1151-4A5F-99B1-058D176352A3.png)
![](Linear%20Regression/2722AA78-2FED-4261-85F5-5829FBCA1F8C.png)
### Overfitting
![](Linear%20Regression/705C19FB-3962-4E1E-9E19-7739F4275659.png)
# Support Vector Machine
## General
* Solve **Classification Problem**
* Input **Feature space**
* Output
binary: Positive vs. Negative 
(anyway it should be discrete)
* Prediction: y = -1/1(f(x) >= 1)

## Loss Function
how to find good parameter values?
### Least square loss
[image:884D9F96-3725-4903-AA64-429E271A13FA-384-00005A499E925BC8/24C31EEC-A223-4C09-97F2-15FDB3A7A5BF.png]
Problem: how to measure mis-classification?
### 0-1 Loss
[image:536EB122-D948-4E46-919E-E647D800F813-384-00005A577E01FBFB/1E842A08-3B48-412C-9DB3-8A38B628861B.png]
Problem: derivative is always 0, not a convex function. 0 feedback.
### Hinge Loss
[image:6769946D-8106-4B2D-AE8E-B76E75D783C2-3339-00003D0C7BBA5043/DD39DDC5-8A66-4CC4-A63E-00C5C7162549.png]
conves: easy to optimize
[image:35483718-8219-4CC5-9EA6-E698BC8C052D-3339-00003D39D9E5DE1F/1F970AD6-A78D-4AC2-8DDC-DE2CB414024E.png]
only points in grey zone will be counted.
Problem: multiple solution? —> prefer smaller w
## Regularization
prefer smaller w
[image:EB5F5F8A-1CE1-4ABD-9450-868A1E992310-384-00005AF029BA2A3A/D88C490F-F437-4EF0-9232-AA0603AAC7E6.png]
Why large margin?
If we add noise to training data, it’s less likely to cross the decision boundary with a larger margin
## Train
Minimize Hinge Loss + Maximize Margin
[image:118F983F-0DBD-4944-9682-F55DA28DA90B-384-00005B74600B108C/B591741F-C9C3-4B78-ACF8-4496F6247D27.png]
When C is very small, prefer maximize margin
When C is ver large, prefer minimize hinge loss
## Support Vector Machine
* Support Vectors = data in the grey zone
* Non-Support Vectors = others (non-support vectors can be deleted without changing the result)
### Sub-gradient Descent 
Introduction to sub-gradient
[Sub-gradient](http://www.hanlongfei.com/convex/2015/10/02/cmu-10725-subgradidient/)
[image:D805863A-980E-42DA-B861-143C171DBC36-384-00005C38963BCB6C/7CE30AA6-51FC-441B-B8EF-AA74B9559BBD.png]
```
initialize w and b
Loop for n_epoch iterations:
	Loop for each training instance (x,y) in trainig set
		compute subgradient
			partial L / partial w
			partial L / partial b
		update the parameters w and b
			w = w - a * partial L / partial w
			b = b - a * partial L / partial b
( a is a constant called learning rate)
```
codes:
[file:F235748A-BAB1-4284-9C48-598DE82E6EF5-384-00005CB56E1CB4CF/problem2.py]
### Dual Problem of  SVM
[image:E316702F-D688-4F44-AE74-4E6EE8565EB5-384-00005D5E11B9505D/5EEBDC05-767D-43B0-B6C9-55E7C981E9F7.png]
finding “ support vectors” directly
[image:F2AB934F-1966-48BE-BA45-48EA394C08A9-384-00005CE388B2D3C5/8B7D0B98-3F7E-4275-9B23-FFE347A79566.png]
Prediction: 
[image:EC7F32E2-368D-4F33-9C4A-80A8C8636D76-384-00005D44F0D351D9/26F69934-FCAB-43CF-9D95-CB31FC25D96F.png]
constraints on SVM weights a:
* 
[image:D3ED26D8-099F-48A0-9738-0E1B862B63D1-384-00005D4F0CBF1DB9/639C71BF-2E85-45A3-AF2F-D7E1AE6A1979.png]
* 
[image:C3141573-932F-4D64-A05F-9171D5A16736-384-00005D524F8A797C/A197BBB6-8ED6-4B9F-89A0-438956A27903.png]
---
**Dual Problem**
[image:8594E727-0FDE-4D8E-A9EB-051F571284D8-384-00005F3F28B3008D/201103131235108008.png]
satisfy KKT condition, then d* = p*
[image:4C83B943-1E47-4539-9473-B9CE2B50A6E9-384-00005F50196071D6/201103131235189201.png]
The problem:
[image:1942BE71-EB2E-402C-89FD-356C656237AF-384-00005F53C3DDFC8B/201103131235281633.png](same with our goal)
[image:8916ED41-210B-4EA3-B219-2A8DCB5ACABA-384-00005F55015F49FA/201103131235316385.png]
then:
1. min part
[image:AD109E4D-5742-4BF3-AD8B-02ABBAE0D435-384-00005F63C1221B91/201103131235339186.png]  [image:FD260E06-9D53-4917-BF5D-DDEBB995FDD7-384-00005F6FBA1904C8/201103131235346579.png]
[image:2F516AB8-C5E5-47A6-BA94-EF30488D7259-384-00005F647546F79A/201103131235346088.png] ( the constraint 2)
[image:3F23449E-3626-4E1E-B35E-1BB64672B3EB-384-00005F7A3022AE97/2011051016205860.png]
[image:6669604C-D583-496E-94ED-6AE5228D8FCA-384-00005F7E96B96653/201103131235354876.png]
the last one is 0:
[image:23934AAF-DA38-436D-B406-C00DE94185E6-384-00005F7CAA037EC4/201103131235352302.png]
2. max part
[image:F72F192F-C637-4E5C-B112-A958780EC154-384-00005F8E6AAB2FE7/201103131235401316.png]
----
[image:67B61F18-F016-4B51-A853-3EF7B52194A7-384-00005FB24D45362E/D859962D-2066-4A0C-BE03-EC083B3FB2AA.png]
### Coordinate Ascent
Each time, 
		only optimize over one variable ai
		while fixing all other variables ak k!=i
### SMO ( Sequential Minimal Optimization)
Each time, 
		only optimize over a **pair**  (ai, aj) of variables
		while fixing all other variables ak  k!=i, k!=j
[image:F80CFB6C-E59F-47FE-B36A-C0F5454F999D-384-00005FF7D66E2F29/54D61726-5521-4506-8159-DD17DC3A8CCE.png]
1. Compute HL(range of ai)
[image:06B3C0DD-E0F8-4EB6-BE49-F9B7E2F2C667-384-000060322B3E2B84/8D880CC3-434E-40D6-BDD1-6E656F35CFC1.png]
[image:797C88D1-22D1-4AA3-9784-CCFF666BD066-384-00006047CE73EE8A/2920CCC8-240B-4209-B0C8-25B3A71EDEF6.png]
[image:17532D24-DC4D-4679-8072-4E440E7C406D-384-00006062C0100907/2D653E8C-BC75-47A5-931E-A011953E4514.png]
[image:CFE6EE59-6159-4156-B90A-AC3A8F804636-384-000060671EC0CEE9/EEE5056B-5DC8-4AC4-ACEF-F016F5862826.png]
2. compute Ei, Ej
[image:4A3F2FF3-8A08-48DB-B611-738F23DFBC48-384-0000608C1464834E/C42BDEAC-6D4D-4923-B8CF-E54ACE8E64FB.png]
3. compute eta  
[image:36F64386-B055-4CA2-8897-B5DB12E1C9E0-384-0000616A6B8DA262/3D6BD55F-B2F9-4C28-AF43-5F9BC0B1295C.png]
4. update ai
[image:3B673C15-D3ED-4D8A-86A4-987283F5CF26-384-00006183593A74C8/357F7228-2C53-4B90-AAED-09484BE8DFA5.png]
5. update aj
[image:FDE731FB-B520-4CF2-9AA1-6547B0FA8B8E-384-000061982F4A928A/217D4400-6CE5-4DEB-BEBA-4524D6EEF1CA.png]
6. update b
[image:19B5CC0D-759A-49B1-9FEB-C060ED9404B4-384-00006203ED5B7552/C2E2EB55-9BF8-4DC2-8985-8AEEA768878A.png]
```
initialize a and b as all zeros
Loops for n_epoch iterations:
	Loops for each pair of instances(i,j) in training set:
		compute range of ai : H,L
		if H == L:
			continue
		compute Ei,Ej,eta
		compute ai_new, aj_new
		update the parameters ai, aj and b
```

## Non-linear
Kernel Trick
[image:B4547DC8-2826-4472-B771-131025D10199-384-0000625A053D6D21/4CA9704B-2D5D-444F-BE1A-F321C782CC1E.png]
[file:FA6FE465-290D-4727-BBF4-0F2833701831-384-000065F5D8147426/problem3.py]


#CS539 Machine Learning/2_Support Vector Machine#
