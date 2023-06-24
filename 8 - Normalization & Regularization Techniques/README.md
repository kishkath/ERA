**Session8: Normalization Techniques** - Describing about why there is a need for normalization, standardization, and different types of normalization such as Layer Normalization, Group Normalization, Batch Normalization.

**Holla, Lets get more neighbours. Lets Scale the data and get more neighbours(pixels) with in a small range of distance.**

   ![download](https://user-images.githubusercontent.com/60026221/215568087-a5a603c3-d4e9-4641-ae21-095f08020d35.jpeg)

    - Generally, this is how normalization/standardization really works.

So, as you may got atleast a small idea what & why scaling is used. The AI world really deals with data, here lets consider the numerical values if we dig deeper the higher the numeric values higher is the computation power higher the usage of memory. 

   * For example: 17*6 = 102 , an easy computation whereas 17772324*6 = needs the energy/power to provide the result. 

Same here applies for memory & computation & networks, larger the data larger the work needed. Lets move to DNN's, here we might face a problem of gradient explosion because of larger values, so to avoid this risks we will scale the data and get them in a range of (-1,1) which is a standard range. 

In DNN's, we perform Batch-Normalization which meant scaling & shifting the values, as there are other normalization techniques such as Layer Normalization, Group Normalization, Instance Normalization.**The current folder of session 5 deals with these techniques of normalization, where and why and how of the techniques** 

**Important Point**: In DNN's, Normalization mean normalizing and standardizing (data with a zero mean & 1 std)

**Batch-Normalization**: Normalization is performed across a **batch of BATCHSIZE** across all layers.

**Layer-Normalization**: Normalization is performed on a layer at a time, whereas in Batch-normalization single part of each layer is performed.

**Group-NOrmalization**: Normalization is performed across the channels by dividing in to groups and normalizes the features within each group. 

![images](https://user-images.githubusercontent.com/60026221/215571530-ede0ccd5-51f3-4472-979f-3abc12c2edc6.jpeg)

The session-8 Folder contains the MNIST model ran with Batch-normalization(incl. L1 Regularization) & Layer-normalization & Group-Normalization.

tree /F

      Results:
    
      Top Accuracies of usage of Normalization Techniques:

      * BN: 76.35

      * GN: 64.76

      * LN: 75.25
      
      ---------------------------------------------------------------------------------------------------------------------------------------------

      Analysis:

      Here, LN & GN has performed a quite good, whereas BN + L1 Reg dipped in the performance as compared to them. GN can be considered as a competitive to BN.

      Training parameters didnt bothered much , while the test plots resulted in different ways for BN & GN as shown below: 

Graphs: 

                      Training Loss,                                   Testing Loss,
                      Below One is Training Accuracy                   Below one is Test-Accuracy.
                      
![plots_S8](https://github.com/kishkath/ERA/assets/60026221/4ce456c5-9f5f-499b-8665-0feaaaba0527)




Mis-Classified Images:


![mis_classified_s8](https://github.com/kishkath/ERA/assets/60026221/8dd10800-3bd7-490e-8d01-8975bf05f5a2)

