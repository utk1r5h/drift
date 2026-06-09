import pandas as pd 
import numpy as np

np.random.seed(42)

n=1000

df1= pd.DataFrame({
  "age": np.random.randint(18, 65, n),
  "salary": np.random.randint(10000,500000, n),
  "experience": np.random.randint(0,20,n)

}
)

df2 = pd.DataFrame({
  "age": np.random.randint(30, 65, n), # age drift 
  "salary": np.random.randint(30000,500000, n), # little bit of salary drift
  "experience": np.random.randint(0,20,n)

})

df1.to_csv("data/reference.csv", index=False)
df2.to_csv("data/current.csv", index=False)



# threshold changes due to built in Bonferroni correction


