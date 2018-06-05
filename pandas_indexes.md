
# Pandas Indexes
Pandas indexes are the most confusing thing about pandas. So let's try to understand it. 
What are the advantages of using indices instead of just storing its' values in columns ?


```python
import pandas as pd
```


```python
drinks = pd.read_csv('http://bit.ly/drinksbycountry')
```


```python
drinks.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>89</td>
      <td>132</td>
      <td>54</td>
      <td>4.9</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>25</td>
      <td>0</td>
      <td>14</td>
      <td>0.7</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>245</td>
      <td>138</td>
      <td>312</td>
      <td>12.4</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>217</td>
      <td>57</td>
      <td>45</td>
      <td>5.9</td>
      <td>Africa</td>
    </tr>
  </tbody>
</table>
</div>




```python
drinks.index
```




    RangeIndex(start=0, stop=193, step=1)



> Note: Every dataframe has an index attribute and a columns attribute

> Note: Indexes are sometimes called the row-labels.


```python
drinks.columns
```




    Index(['country', 'beer_servings', 'spirit_servings', 'wine_servings',
           'total_litres_of_pure_alcohol', 'continent'],
          dtype='object')



Index is type of special object, it is NOT refered to as "THE INDEX" though. When someone says "THE INDEX" or the row labels, they are talking about the row indexes.

> Note: Neither the Index or the Columns are considered part of the dataframe contents. This can be seen when you run the shape command.


```python
drinks.shape
```




    (193, 6)



It turns out that both these Indexes and columns both default to integers. That is, if no index or columns are specified. For example, take a look at the movieusers dataframe


```python
pd.read_table('http://bit.ly/movieusers', header=None, sep='|').head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>85711</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>53</td>
      <td>F</td>
      <td>other</td>
      <td>94043</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>23</td>
      <td>M</td>
      <td>writer</td>
      <td>32067</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>43537</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>33</td>
      <td>F</td>
      <td>other</td>
      <td>15213</td>
    </tr>
  </tbody>
</table>
</div>



As we did not specify the header, the header and the row indexes are set to default integer values.

> Note: Sometimes people just leave the default values for row indexes, but rarely they leave the default values for column names.

Coming back to our main question: Why does the Index exist ?
    There are 3 main reasons:
    - One is identification
    - Second is selection
    - Third is alignment

For identification, we are going to illustrate this by taking an example. We are going to do filtering the dataframe.


```python
drinks[drinks.continent=='South America']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Argentina</td>
      <td>193</td>
      <td>25</td>
      <td>221</td>
      <td>8.3</td>
      <td>South America</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Bolivia</td>
      <td>167</td>
      <td>41</td>
      <td>8</td>
      <td>3.8</td>
      <td>South America</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Brazil</td>
      <td>245</td>
      <td>145</td>
      <td>16</td>
      <td>7.2</td>
      <td>South America</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Chile</td>
      <td>130</td>
      <td>124</td>
      <td>172</td>
      <td>7.6</td>
      <td>South America</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Colombia</td>
      <td>159</td>
      <td>76</td>
      <td>3</td>
      <td>4.2</td>
      <td>South America</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Ecuador</td>
      <td>162</td>
      <td>74</td>
      <td>3</td>
      <td>4.2</td>
      <td>South America</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Guyana</td>
      <td>93</td>
      <td>302</td>
      <td>1</td>
      <td>7.1</td>
      <td>South America</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Paraguay</td>
      <td>213</td>
      <td>117</td>
      <td>74</td>
      <td>7.3</td>
      <td>South America</td>
    </tr>
    <tr>
      <th>133</th>
      <td>Peru</td>
      <td>163</td>
      <td>160</td>
      <td>21</td>
      <td>6.1</td>
      <td>South America</td>
    </tr>
    <tr>
      <th>163</th>
      <td>Suriname</td>
      <td>128</td>
      <td>178</td>
      <td>7</td>
      <td>5.6</td>
      <td>South America</td>
    </tr>
    <tr>
      <th>185</th>
      <td>Uruguay</td>
      <td>115</td>
      <td>35</td>
      <td>220</td>
      <td>6.6</td>
      <td>South America</td>
    </tr>
    <tr>
      <th>188</th>
      <td>Venezuela</td>
      <td>333</td>
      <td>100</td>
      <td>3</td>
      <td>7.7</td>
      <td>South America</td>
    </tr>
  </tbody>
</table>
</div>



The thing I want you to notice is that the Index (also known as the row labels), STAYED with the rows. It didn't just re-number them as starting from 0. It kept the original row numbers. This is what we mean by, when we say that the Index is for "Identification". It is so that we can identify what rows we are working with even if you filter the original data frame.

Next, let's talk about "Selection". And what I mean by this, "what if I want to grab a piece of this dataframe ?" . We are going to use our beloved `loc` method to do that. The `loc` method, allows me say "if I want number 245 in the above dataframe" then I can simply use the below line


```python
drinks.loc[23, 'beer_servings']
```




    245



So what's the big deal here ? Why bother having an index ? Here's the answer to it. Let me show you.


```python
# We can set the index using set_index dataframe method
# When you set inplace=True, then you don't have to assign it to a variable. It does it inplace.
drinks.set_index('country', inplace=True)
drinks.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
      <th>continent</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>89</td>
      <td>132</td>
      <td>54</td>
      <td>4.9</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>25</td>
      <td>0</td>
      <td>14</td>
      <td>0.7</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>245</td>
      <td>138</td>
      <td>312</td>
      <td>12.4</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>217</td>
      <td>57</td>
      <td>45</td>
      <td>5.9</td>
      <td>Africa</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the dataframe has changed. We can see above that the Series country has now become an index and the default (integer index) has disappeared.


```python
drinks.index
```




    Index(['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola',
           'Antigua & Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria',
           ...
           'Tanzania', 'USA', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela',
           'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe'],
          dtype='object', name='country', length=193)



Few things to observe here:
- It still has length 193
- It is of type 'object', which by the way is the same as default index, since it is an Index object.
- It has a new attribute called 'name'


```python
drinks.columns
```




    Index(['beer_servings', 'spirit_servings', 'wine_servings',
           'total_litres_of_pure_alcohol', 'continent'],
          dtype='object')



Also note now that country is no longer listed as a column in the dataframe.


```python
drinks.shape
```




    (193, 5)



Also check the shape it now has 193 and only 5 columns. This is because the Index is now not part of the dataframe. Remember that.

So, because we have set the country as the index, we can now use our beloved `loc` method to do something like this.


```python
drinks.loc['Brazil', 'beer_servings']
```




    245



### So, by setting that index to something that was meaningful to us, we can now select data from the dataframe more easily!!! 

Now, couple of issues to discuss, if you look at the dataframe output, you will notice that "country" is not listed in the columns row, it has a new separate row all for itself. Looks weird right ? That is actually the *name of the index*.

Now, you don't need to have an index name. It is helpful, in that, it can serve as an identifier, that is it represents countries. But you can actually clear it out, if you so desire.


```python
drinks.index.name = None
drinks.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>89</td>
      <td>132</td>
      <td>54</td>
      <td>4.9</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>25</td>
      <td>0</td>
      <td>14</td>
      <td>0.7</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>245</td>
      <td>138</td>
      <td>312</td>
      <td>12.4</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>217</td>
      <td>57</td>
      <td>45</td>
      <td>5.9</td>
      <td>Africa</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's say you decide at some point that you want to use the default index and now you want to move that index (country index) back in as a column, then you do this.

Firstly, we need to give that index its' name back!


```python
drinks.index.name = 'country'
drinks.reset_index(inplace=True)
drinks.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>89</td>
      <td>132</td>
      <td>54</td>
      <td>4.9</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>25</td>
      <td>0</td>
      <td>14</td>
      <td>0.7</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>245</td>
      <td>138</td>
      <td>312</td>
      <td>12.4</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>217</td>
      <td>57</td>
      <td>45</td>
      <td>5.9</td>
      <td>Africa</td>
    </tr>
  </tbody>
</table>
</div>



And if you now check this out, we are back to our default integer index and the country which was the index, rejoined the dataframe as one of the columns. It is important to set the index name before we re-index, because pandas need to know what to call the new column.

Let's do something extra. When we run the `describe` method, see below


```python
drinks.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>193.000000</td>
      <td>193.000000</td>
      <td>193.000000</td>
      <td>193.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>106.160622</td>
      <td>80.994819</td>
      <td>49.450777</td>
      <td>4.717098</td>
    </tr>
    <tr>
      <th>std</th>
      <td>101.143103</td>
      <td>88.284312</td>
      <td>79.697598</td>
      <td>3.773298</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>76.000000</td>
      <td>56.000000</td>
      <td>8.000000</td>
      <td>4.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>188.000000</td>
      <td>128.000000</td>
      <td>59.000000</td>
      <td>7.200000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>376.000000</td>
      <td>438.000000</td>
      <td>370.000000</td>
      <td>14.400000</td>
    </tr>
  </tbody>
</table>
</div>



It is just a numerical summary of the numerical columns. And, I want you to notice that this is actually a dataframe. And as such, it has an index.


```python
drinks.describe().index
```




    Index(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], dtype='object')



This is the index!!

And these are the columns


```python
drinks.describe().columns
```




    Index(['beer_servings', 'spirit_servings', 'wine_servings',
           'total_litres_of_pure_alcohol'],
          dtype='object')




```python
drinks.describe().loc[['mean','count'],:]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>106.160622</td>
      <td>80.994819</td>
      <td>49.450777</td>
      <td>4.717098</td>
    </tr>
    <tr>
      <th>count</th>
      <td>193.000000</td>
      <td>193.000000</td>
      <td>193.000000</td>
      <td>193.000000</td>
    </tr>
  </tbody>
</table>
</div>



So, the point here is not so much that you are going to do something with the describe output! You might, like above. But the main point here is that, a lot of methods in pandas return a dataframe. And when you know that there is something like a index and columns associated with that dataframe, and you recognize that, then you can interact with that resulting dataframe in cool ways.

### A SERIES ALSO HAS AN INDEX

The last topic is "Arrangement". What I want to show you, is that, a Series like a Dataframe also has an Index.

The Series index comes from the dataframe!!


```python
drinks.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>89</td>
      <td>132</td>
      <td>54</td>
      <td>4.9</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>25</td>
      <td>0</td>
      <td>14</td>
      <td>0.7</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>245</td>
      <td>138</td>
      <td>312</td>
      <td>12.4</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>217</td>
      <td>57</td>
      <td>45</td>
      <td>5.9</td>
      <td>Africa</td>
    </tr>
  </tbody>
</table>
</div>



Let's say now I pick the continent series.


```python
drinks.continent.head()
```




    0      Asia
    1    Europe
    2    Africa
    3    Europe
    4    Africa
    Name: continent, dtype: object



What we see here is that there is an Index for this Series and it came from the dataframe.

So the Index is on the left and the values are on the right.

Now, let's pretend that we didn't use the default index for the dataframe, and instead, we use the country as the index for the dataframe.


```python
drinks.set_index('country', inplace=True)
drinks.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
      <th>continent</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>89</td>
      <td>132</td>
      <td>54</td>
      <td>4.9</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>25</td>
      <td>0</td>
      <td>14</td>
      <td>0.7</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>245</td>
      <td>138</td>
      <td>312</td>
      <td>12.4</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>217</td>
      <td>57</td>
      <td>45</td>
      <td>5.9</td>
      <td>Africa</td>
    </tr>
  </tbody>
</table>
</div>



SO, what is going to happen, if we are going to select the continent Series ??


```python
drinks.continent.head()
```




    country
    Afghanistan      Asia
    Albania        Europe
    Algeria        Africa
    Andorra        Europe
    Angola         Africa
    Name: continent, dtype: object



This time, we are seeing the same thing as last time, the INDEX is passed on from the dataframe. The index is attached to each row!


```python
drinks.continent.value_counts()
```




    Africa           53
    Europe           45
    Asia             44
    North America    23
    Oceania          16
    South America    12
    Name: continent, dtype: int64



This is also a Series, and as such, it has an index.


```python
drinks.continent.value_counts().index
```




    Index(['Africa', 'Europe', 'Asia', 'North America', 'Oceania',
           'South America'],
          dtype='object')



And it has VALUES (to the right remember).


```python
drinks.continent.value_counts().values
```




    array([53, 45, 44, 23, 16, 12])



NOW because it is a SERIES, and not some value_counts object or something, we can use the index to select values from the Series. What I mean is, I can take this `values` array and refer an index, such as 'Africa'.


```python
drinks.continent.value_counts()['Africa']
```




    53



What I am saying is: "From this Series find index Africa and show me the value"

## Sorting


```python
drinks.continent.value_counts().sort_values()
```




    South America    12
    Oceania          16
    North America    23
    Asia             44
    Europe           45
    Africa           53
    Name: continent, dtype: int64



Using the `sort_values` method of the Series object, we can sort the **VALUES** of the Series.

WHAT IF YOU WANTED TO SORT ON THE INDEX ??


```python
drinks.continent.value_counts().sort_index()
```




    Africa           53
    Asia             44
    Europe           45
    North America    23
    Oceania          16
    South America    12
    Name: continent, dtype: int64



Remember the 3 reasons why an Index exists:
- Identification
- Selection
- Allignment

## Allignment

In order to understand alignment, we first need to create another Series object.


```python
people = pd.Series([300000, 85000], index=['Albania', 'Andorra'], name='population')
people
```




    Albania    300000
    Andorra     85000
    Name: population, dtype: int64



Let's say I want to use this tiny dataset and multiply this with `drinks.beer_servings` to get the total number of people who were served beer.


```python
drinks.beer_servings * people
```




    Afghanistan                    NaN
    Albania                 26700000.0
    Algeria                        NaN
    Andorra                 20825000.0
    Angola                         NaN
    Antigua & Barbuda              NaN
    Argentina                      NaN
    Armenia                        NaN
    Australia                      NaN
    Austria                        NaN
    Azerbaijan                     NaN
    Bahamas                        NaN
    Bahrain                        NaN
    Bangladesh                     NaN
    Barbados                       NaN
    Belarus                        NaN
    Belgium                        NaN
    Belize                         NaN
    Benin                          NaN
    Bhutan                         NaN
    Bolivia                        NaN
    Bosnia-Herzegovina             NaN
    Botswana                       NaN
    Brazil                         NaN
    Brunei                         NaN
    Bulgaria                       NaN
    Burkina Faso                   NaN
    Burundi                        NaN
    Cabo Verde                     NaN
    Cambodia                       NaN
                               ...    
    Sudan                          NaN
    Suriname                       NaN
    Swaziland                      NaN
    Sweden                         NaN
    Switzerland                    NaN
    Syria                          NaN
    Tajikistan                     NaN
    Tanzania                       NaN
    Thailand                       NaN
    Timor-Leste                    NaN
    Togo                           NaN
    Tonga                          NaN
    Trinidad & Tobago              NaN
    Tunisia                        NaN
    Turkey                         NaN
    Turkmenistan                   NaN
    Tuvalu                         NaN
    USA                            NaN
    Uganda                         NaN
    Ukraine                        NaN
    United Arab Emirates           NaN
    United Kingdom                 NaN
    Uruguay                        NaN
    Uzbekistan                     NaN
    Vanuatu                        NaN
    Venezuela                      NaN
    Vietnam                        NaN
    Yemen                          NaN
    Zambia                         NaN
    Zimbabwe                       NaN
    dtype: float64



The countries not represented in the Series: people are ignored. And only the countries represented in the people series are multiplied. So how does pandas do this ? So here's the thing, it **aligned** the `people` series with the drink series. That is, it found Albania in the `drink` series and then multiplied by the population value in the `people` series.

So, in summary, **alignment** allows us to put the data together even if it not the same length. As long as you tell it, which rows correspond to which other rows. Because `people` and `drinks` series use country as the index and since we specified the index in `people`, pandas used that index to match the rows in `drinks` series.

One more example to hammer the point home. Suppose lets say you want to add the people Series to the drinks dataframe as a new column. To do this, we use the pd.concat method.


```python
pd.concat([drinks, people], axis=1).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
      <th>continent</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>Asia</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>89</td>
      <td>132</td>
      <td>54</td>
      <td>4.9</td>
      <td>Europe</td>
      <td>300000.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>25</td>
      <td>0</td>
      <td>14</td>
      <td>0.7</td>
      <td>Africa</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>245</td>
      <td>138</td>
      <td>312</td>
      <td>12.4</td>
      <td>Europe</td>
      <td>85000.0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>217</td>
      <td>57</td>
      <td>45</td>
      <td>5.9</td>
      <td>Africa</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Notice how pandas automatically knew how to align the values in the merged dataframe. It does so, because of the index. Without the index, pandas wouldn't know where to put them. So that is the beauty and elegance of having an index. Hope that explains it.
