# Modeling Asteroid Impact based on its Features



Using "Period.Start" in one analysis and "Period.End" in another can be strategic and intentional, based on the specific insights or trends you are aiming to uncover within your dataset, especially in contexts like asteroid impact risk analysis. Here's why differentiating between these two might be meaningful:

### Period.End for Counting Asteroid Risks

When grouping by "Period.End" to count asteroid risks, the focus is on understanding when the risk of asteroid impacts concludes within a given period. This approach might be particularly useful for assessing:

- **Risk Mitigation Strategies:** Identifying when certain risks are expected to subside can help in planning and prioritizing risk mitigation strategies. It's about knowing by when certain protective measures need to be in place.
- **Impact Forecasting:** Forecasting the end period of potential asteroid impacts can provide insights into the temporal distribution of these events, helping to understand when the Earth is more susceptible to experiencing these events.

### Period.Start for Cumulative Impact Probability

Conversely, analyzing "Period.Start" for the average cumulative impact probability focuses on the initiation of risk periods, which can offer different insights:

- **Early Warning Systems:** Emphasizing the start of a period can be crucial for early warning systems and preparedness. Knowing when the probability of impacts begins to rise allows for preemptive actions.
- **Policy and Planning:** For policy-making and planning, understanding the onset of risk periods can inform the development of guidelines and the allocation of resources well in advance of potential impacts.

### Strategic Importance

The strategic use of "Period.Start" and "Period.End" caters to different analytical needs and decision-making processes:

- **Temporal Analysis:** It allows for a nuanced temporal analysis of asteroid impact risks, differentiating between the onset of risk and its conclusion.
- **Decision-making:** Different stakeholders may require information specific to the beginning or end of risk periods. For example, emergency services might be more interested in "Period.Start" for immediate response planning, while long-term urban or community planning might focus more on "Period.End" to understand future risk landscapes.

In summary, the choice between "Period.Start" and "Period.End" hinges on the specific objectives of the analysis, the nature of the decision-making process it informs, and the temporal insights required to effectively manage or mitigate asteroid impact risks.

Results of models without any feature engineered data:

                                      MSE            R2
Linear Regression            2.149896e+03  2.640356e-01
KNN                          2.380218e+03  1.851905e-01
Stochastic Gradient Descent  5.908724e+21 -2.022708e+18
DecisionTreeClassifier       6.388195e+03 -1.186843e+00
Random Forest                1.824739e+03  3.753451e-01
Gradient Boosting            2.176319e+03  2.549903e-01
Ridge Regression             2.149861e+03  2.640474e-01

Results of Poly Scaled data:

                                      MSE            R2
Linear Regression            4.167147e+03 -4.265213e-01
KNN                          2.579562e+03  1.169498e-01
Stochastic Gradient Descent  2.166098e+11 -7.415107e+07
DecisionTreeClassifier       2.298213e+03  2.132629e-01
Random Forest                2.372833e+03  1.877185e-01
Gradient Boosting            3.488175e+03 -1.940915e-01
Ridge Regression             2.297630e+03  2.134622e-01

Results of RFE selected data:

                                      MSE            R2
Linear Regression            2.325755e+03  2.038346e-01
KNN                          3.512773e+03 -2.025123e-01
Stochastic Gradient Descent  1.001282e+17 -3.427646e+13
DecisionTreeClassifier       8.232878e+03 -1.818325e+00
Random Forest                2.503413e+03  1.430177e-01
Gradient Boosting            2.314810e+03  2.075811e-01
Ridge Regression             2.325714e+03  2.038484e-01


Results of Scaled data:

                                     MSE        R2
Linear Regression            2149.895640  0.264036
KNN                          2286.995102  0.217103
Stochastic Gradient Descent  2150.063564  0.263978
DecisionTreeClassifier       2049.128280  0.298531
Random Forest                1887.934643  0.353712
Gradient Boosting            2201.576138  0.246344
Ridge Regression             2149.861321  0.264047

