
## What are different ways in which you can create a dataframe ?
- Using pd.DataFrame() constructor.
- Zip lists to build a DataFrame.
- Building dataframes with broadcasting.

### Using pd.DataFrame() constructor
We can create a dataframe by using the `pd.DataFrame()` constructor. The dictionary-list constructor assigns values to the column labels, but just uses an ascending count from 0 (0, 1, 2, 3, ...) for the row labels. 


```python
# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Import pandas as pd
import pandas as pd

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }

# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

# Print cars
print(cars)
```

       cars_per_cap        country  drives_right
    0           809  United States          True
    1           731      Australia         False
    2           588          Japan         False
    3            18          India         False
    4           200         Russia          True
    5            70        Morocco          True
    6            45          Egypt          True


Another simple example is shown below. Here we are using the same way as above - i.e, using a dictionary of lists to create a dataframe. In addition, we also show below how we can modify the index of the rows after creating the dataframe.


```python
# Your code here
Apples = [35, 41]
Bananas = [21, 34]
rows = ['2017 Sales', '2018 Sales']

df = pd.DataFrame({'Apples': Apples, 'Bananas': Bananas})
df.index = rows
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apples</th>
      <th>Bananas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017 Sales</th>
      <td>35</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2018 Sales</th>
      <td>41</td>
      <td>34</td>
    </tr>
  </tbody>
</table>
</div>



Instead of creating the index after creating the dataframe, there is another way to create the index while creating the dataframe itself.


```python
# Your code here
Apples = [35, 41]
Bananas = [21, 34]
rows = ['2017 Sales', '2018 Sales']

df = pd.DataFrame({'Apples': Apples, 'Bananas': Bananas}, index=rows)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apples</th>
      <th>Bananas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017 Sales</th>
      <td>35</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2018 Sales</th>
      <td>41</td>
      <td>34</td>
    </tr>
  </tbody>
</table>
</div>



### Using Zip Lists to build DataFrames
Suppose you have `list_keys` and `list_values` as two seperate lists, and you want to use the `list_keys` as the column names and `list_values` as the column data, then you can use the `zip` and `list` functions to create the desired dataframes.


```python
list_keys = ['Country', 'GDP']
list_values = [['US', 'AUS', 'IND'], [400, 300, 100]]
zipped_lists = zip(list_keys, list_values)

# Zip object - a generator
print(zipped_lists)

# convert to a list of tuples
zipped_tuples = list(zipped_lists)

# list of tuples 
print(zipped_tuples)
```

    <zip object at 0x11493ae08>
    [('Country', ['US', 'AUS', 'IND']), ('GDP', [400, 300, 100])]


Using the above list of tuples, we can convert them into a dictionary. Remember, pd.DataFrame() takes a dict object.


```python
# convert to a dictionary
dict_lists = dict(zipped_tuples)

print(dict_lists)

```

    {'Country': ['US', 'AUS', 'IND'], 'GDP': [400, 300, 100]}



```python
# Finally create a dataframe.
df1 = pd.DataFrame(dict_lists)

# print
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US</td>
      <td>400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AUS</td>
      <td>300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IND</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



### Using broadcasting
You can implicitly use 'broadcasting', a feature of NumPy, when creating pandas DataFrames. For example, here we  create a DataFrame of cities in Pennsylvania that contains the city name in one column and the state name in the second. 


```python
cities = ['Manheim',
 'Preston park',
 'Biglerville',
 'Indiana',
 'Curwensville',
 'Crown',
 'Harveys lake',
 'Mineral springs',
 'Cassville',
 'Hannastown',
 'Saltsburg',
 'Tunkhannock',
 'Pittsburgh',
 'Lemasters',
 'Great bend']

# Make a string with the value 'PA': state
state = 'PA'

# Construct a dictionary: data
data = {'state':state, 'city':cities}

# Construct a DataFrame from dictionary data: df
df = pd.DataFrame(data)

# Print the DataFrame
print(df)
```

                   city state
    0           Manheim    PA
    1      Preston park    PA
    2       Biglerville    PA
    3           Indiana    PA
    4      Curwensville    PA
    5             Crown    PA
    6      Harveys lake    PA
    7   Mineral springs    PA
    8         Cassville    PA
    9        Hannastown    PA
    10        Saltsburg    PA
    11      Tunkhannock    PA
    12       Pittsburgh    PA
    13        Lemasters    PA
    14       Great bend    PA

