The simulate-and-recover process for the EZ Diffusion model was conducted to test whether the model could accurately estimate three key parameters: boundary separation (α), drift rate (ν), and non-decision time (τ). The main goal was to check if the bias (difference between estimated and true values) converges to zero and whether the squared error (variability in estimates) decreases as the sample size increases. The results confirm that the model behaves as expected: more data leads to more accurate and stable parameter recovery.

- Bias measures how far the recovered parameters deviate from the actual values. Ideally, if the model is working correctly, bias should approach zero as N increases.
   - N = 10 → Bias values are still noticeable, particularly for boundary separation (0.2178). This suggests that with small sample sizes, there is still some systematic error in parameter recovery.
   - N = 40 → Bias values drop significantly, with boundary separation bias at 0.0231 and drift rate bias at -0.0208. This indicates that increasing sample size helps the model stabilize.
   - N = 4000 → Bias values are practically zero (0.0005 for drift rate, 0.0006 for boundary separation), confirming that the model does not systematically overestimate or underestimate any parameters when given sufficient data.
Overall, the decreasing trend in bias means that the EZ Diffusion model is statistically unbiased when tested across large sample sizes.

- Squared error measures the precision of the estimates—how much they fluctuate from one trial to another. A good model should show lower squared errors as N increases, meaning the recovered parameters become more stable.
   - N = 10 → Squared errors are relatively high (0.4078 for drift rate, 0.2013 for boundary separation), indicating significant variability in estimates.
   - N = 40 → Squared errors drop considerably (0.1202 for drift rate, 0.0420 for boundary separation), meaning that as more data is introduced, estimates become more consistent.
   - N = 4000 → Squared errors approach zero (0.0012 for drift rate, 0.0001 for boundary separation), confirming that with large datasets, the model achieves high precision in parameter recovery.
The trend is: with more data, the model produces more stable and reliable estimates. This follows standard statistical principles—small sample sizes introduce noise, but as N increases, random fluctuations average out.
