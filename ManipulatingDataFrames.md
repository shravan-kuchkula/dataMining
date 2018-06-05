
## Grouping and Aggregating dataframes
The Olympic medal data for the following exercises comes from https://assets.datacamp.com/production/course_1650/datasets/all_medalists.csv . It comprises records of all events held at the Olympic games between 1896 and 2012. Suppose you have loaded the data into a DataFrame medals. You now want to find the total number of medals awarded to the USA per edition.


```python
import pandas as pd
```


```python
medals = pd.read_csv('https://assets.datacamp.com/production/course_1650/datasets/all_medalists.csv')
medals.head()
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
      <th>City</th>
      <th>Edition</th>
      <th>Sport</th>
      <th>Discipline</th>
      <th>Athlete</th>
      <th>NOC</th>
      <th>Gender</th>
      <th>Event</th>
      <th>Event_gender</th>
      <th>Medal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Athens</td>
      <td>1896</td>
      <td>Aquatics</td>
      <td>Swimming</td>
      <td>HAJOS, Alfred</td>
      <td>HUN</td>
      <td>Men</td>
      <td>100m freestyle</td>
      <td>M</td>
      <td>Gold</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Athens</td>
      <td>1896</td>
      <td>Aquatics</td>
      <td>Swimming</td>
      <td>HERSCHMANN, Otto</td>
      <td>AUT</td>
      <td>Men</td>
      <td>100m freestyle</td>
      <td>M</td>
      <td>Silver</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Athens</td>
      <td>1896</td>
      <td>Aquatics</td>
      <td>Swimming</td>
      <td>DRIVAS, Dimitrios</td>
      <td>GRE</td>
      <td>Men</td>
      <td>100m freestyle for sailors</td>
      <td>M</td>
      <td>Bronze</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Athens</td>
      <td>1896</td>
      <td>Aquatics</td>
      <td>Swimming</td>
      <td>MALOKINIS, Ioannis</td>
      <td>GRE</td>
      <td>Men</td>
      <td>100m freestyle for sailors</td>
      <td>M</td>
      <td>Gold</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Athens</td>
      <td>1896</td>
      <td>Aquatics</td>
      <td>Swimming</td>
      <td>CHASAPIS, Spiridon</td>
      <td>GRE</td>
      <td>Men</td>
      <td>100m freestyle for sailors</td>
      <td>M</td>
      <td>Silver</td>
    </tr>
  </tbody>
</table>
</div>




```python
medals.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 29216 entries, 0 to 29215
    Data columns (total 10 columns):
    City            29216 non-null object
    Edition         29216 non-null int64
    Sport           29216 non-null object
    Discipline      29216 non-null object
    Athlete         29216 non-null object
    NOC             29216 non-null object
    Gender          29216 non-null object
    Event           29216 non-null object
    Event_gender    29216 non-null object
    Medal           29216 non-null object
    dtypes: int64(1), object(9)
    memory usage: 2.2+ MB


The columns `NOC` represents the country. 


```python
medals.loc[medals.NOC == "USA"].groupby('Edition')['Medal'].count()
```




    Edition
    1896     20
    1900     55
    1904    394
    1908     63
    1912    101
    1920    193
    1924    198
    1928     84
    1932    181
    1936     92
    1948    148
    1952    130
    1956    118
    1960    112
    1964    150
    1968    149
    1972    155
    1976    155
    1984    333
    1988    193
    1992    224
    1996    260
    2000    248
    2004    264
    2008    315
    Name: Medal, dtype: int64



####  What are the top 15 countries ranked by total number of medals ?

For this, we can use the pandas Series method `.value_counts()` to determine the top 15 countries ranked by total number of medals.


```python
# Select the 'NOC' column of medals: country_names
country_names = medals['NOC']

# Count the number of medals won by each country: medal_counts
medal_counts = country_names.value_counts()

# Print top 15 countries ranked by medals
print(medal_counts.head(15))
```

    USA    4335
    URS    2049
    GBR    1594
    FRA    1314
    ITA    1228
    GER    1211
    AUS    1075
    HUN    1053
    SWE    1021
    GDR     825
    NED     782
    JPN     704
    CHN     679
    RUS     638
    ROU     624
    Name: NOC, dtype: int64


#### Using `.pivot_table()` to count medals by type

Let's see how using `.pivot_table()` helps in summarizing the data. Take a look at the head().


```python
medals.head()
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
      <th>City</th>
      <th>Edition</th>
      <th>Sport</th>
      <th>Discipline</th>
      <th>Athlete</th>
      <th>NOC</th>
      <th>Gender</th>
      <th>Event</th>
      <th>Event_gender</th>
      <th>Medal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Athens</td>
      <td>1896</td>
      <td>Aquatics</td>
      <td>Swimming</td>
      <td>HAJOS, Alfred</td>
      <td>HUN</td>
      <td>Men</td>
      <td>100m freestyle</td>
      <td>M</td>
      <td>Gold</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Athens</td>
      <td>1896</td>
      <td>Aquatics</td>
      <td>Swimming</td>
      <td>HERSCHMANN, Otto</td>
      <td>AUT</td>
      <td>Men</td>
      <td>100m freestyle</td>
      <td>M</td>
      <td>Silver</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Athens</td>
      <td>1896</td>
      <td>Aquatics</td>
      <td>Swimming</td>
      <td>DRIVAS, Dimitrios</td>
      <td>GRE</td>
      <td>Men</td>
      <td>100m freestyle for sailors</td>
      <td>M</td>
      <td>Bronze</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Athens</td>
      <td>1896</td>
      <td>Aquatics</td>
      <td>Swimming</td>
      <td>MALOKINIS, Ioannis</td>
      <td>GRE</td>
      <td>Men</td>
      <td>100m freestyle for sailors</td>
      <td>M</td>
      <td>Gold</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Athens</td>
      <td>1896</td>
      <td>Aquatics</td>
      <td>Swimming</td>
      <td>CHASAPIS, Spiridon</td>
      <td>GRE</td>
      <td>Men</td>
      <td>100m freestyle for sailors</td>
      <td>M</td>
      <td>Silver</td>
    </tr>
  </tbody>
</table>
</div>



`pivot_table` is used when you want to see how a particular column relates to another column. For example, here we want to see how each country (`NOC`) column relates to the `Medal` column. Another example is if you want to know how the medals are distributed amoung men and women for each country, you can do this:


```python
medals.pivot_table(index='NOC', columns='Gender', values = 'Athlete', aggfunc='count')
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
      <th>Gender</th>
      <th>Men</th>
      <th>Women</th>
    </tr>
    <tr>
      <th>NOC</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AFG</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>AHO</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ALG</th>
      <td>11.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>ANZ</th>
      <td>27.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>ARG</th>
      <td>183.0</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>ARM</th>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>AUS</th>
      <td>647.0</td>
      <td>428.0</td>
    </tr>
    <tr>
      <th>AUT</th>
      <td>125.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>AZE</th>
      <td>12.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>BAH</th>
      <td>10.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>BAR</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BDI</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BEL</th>
      <td>390.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>BER</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BLR</th>
      <td>45.0</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>BOH</th>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>262.0</td>
      <td>110.0</td>
    </tr>
    <tr>
      <th>BUL</th>
      <td>214.0</td>
      <td>117.0</td>
    </tr>
    <tr>
      <th>BWI</th>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>411.0</td>
      <td>181.0</td>
    </tr>
    <tr>
      <th>CHI</th>
      <td>32.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>CHN</th>
      <td>218.0</td>
      <td>461.0</td>
    </tr>
    <tr>
      <th>CIV</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CMR</th>
      <td>20.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>COL</th>
      <td>6.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>CRC</th>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>CRO</th>
      <td>75.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>CUB</th>
      <td>299.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>CZE</th>
      <td>29.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>DEN</th>
      <td>402.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>SRI</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>SUD</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SUI</th>
      <td>350.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SVK</th>
      <td>24.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>SWE</th>
      <td>932.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>SYR</th>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>TAN</th>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TCH</th>
      <td>249.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>TGA</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>THA</th>
      <td>13.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>TJK</th>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TOG</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TPE</th>
      <td>30.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>TRI</th>
      <td>20.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TUN</th>
      <td>7.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TUR</th>
      <td>75.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>UAE</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>UGA</th>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>UKR</th>
      <td>75.0</td>
      <td>73.0</td>
    </tr>
    <tr>
      <th>URS</th>
      <td>1476.0</td>
      <td>573.0</td>
    </tr>
    <tr>
      <th>URU</th>
      <td>76.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>3120.0</td>
      <td>1215.0</td>
    </tr>
    <tr>
      <th>UZB</th>
      <td>16.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>9.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>VIE</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>YUG</th>
      <td>373.0</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>ZAM</th>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ZIM</th>
      <td>NaN</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>ZZX</th>
      <td>45.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>138 rows × 2 columns</p>
</div>



Let's say you are now interested in finding out **which countries have women won as many or more medals than men** To answer this question, we need to take the above output and then add another boolean Series/column which is True for a country where women won more medals than men. Then, use that column to filter the dataframe to find the countries where women won more medals.


```python
genderCounts = medals.pivot_table(index='NOC', columns='Gender', values = 'Athlete', aggfunc='count')
genderCounts.loc[genderCounts['Women'] >= genderCounts['Men']]
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
      <th>Gender</th>
      <th>Men</th>
      <th>Women</th>
    </tr>
    <tr>
      <th>NOC</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BAH</th>
      <td>10.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>BLR</th>
      <td>45.0</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>CHN</th>
      <td>218.0</td>
      <td>461.0</td>
    </tr>
    <tr>
      <th>IOP</th>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>47.0</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>3.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>ROU</th>
      <td>299.0</td>
      <td>325.0</td>
    </tr>
    <tr>
      <th>SIN</th>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>SRI</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>VIE</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



While this is true for the countries listed above, there are some countries which have `NAN` values, example `ZIM`. These are lost during comparison. Hence, we must first replace `NAN` values from our `genderCounts` dataframe with 0's.


```python
genderCounts.fillna(value=0, inplace=True)
```

Now, if we run our comparison, we can get the countries where women won as many or more medals than men.


```python
genderCounts.loc[genderCounts['Women'] >= genderCounts['Men']]
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
      <th>Gender</th>
      <th>Men</th>
      <th>Women</th>
    </tr>
    <tr>
      <th>NOC</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BAH</th>
      <td>10.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>BLR</th>
      <td>45.0</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>CHN</th>
      <td>218.0</td>
      <td>461.0</td>
    </tr>
    <tr>
      <th>CRC</th>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>IOP</th>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>47.0</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>MOZ</th>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>3.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>ROU</th>
      <td>299.0</td>
      <td>325.0</td>
    </tr>
    <tr>
      <th>SIN</th>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>SRI</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>VIE</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ZIM</th>
      <td>0.0</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>



Now, we see that there are 3 more countries added to the list, namely, `ZIM`, `MOZ` and `CRC`. 

Next, I am interested in knowing in which countries women won more medals than men ?


```python
medals.pivot_table(index='NOC', columns=['Gender', 'Medal'], values = 'Athlete', aggfunc='count')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Gender</th>
      <th colspan="3" halign="left">Men</th>
      <th colspan="3" halign="left">Women</th>
    </tr>
    <tr>
      <th>Medal</th>
      <th>Bronze</th>
      <th>Gold</th>
      <th>Silver</th>
      <th>Bronze</th>
      <th>Gold</th>
      <th>Silver</th>
    </tr>
    <tr>
      <th>NOC</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AFG</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>AHO</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ALG</th>
      <td>7.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ANZ</th>
      <td>5.0</td>
      <td>19.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ARG</th>
      <td>51.0</td>
      <td>68.0</td>
      <td>64.0</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>ARM</th>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>AUS</th>
      <td>263.0</td>
      <td>148.0</td>
      <td>236.0</td>
      <td>150.0</td>
      <td>145.0</td>
      <td>133.0</td>
    </tr>
    <tr>
      <th>AUT</th>
      <td>31.0</td>
      <td>17.0</td>
      <td>77.0</td>
      <td>13.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>AZE</th>
      <td>6.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAH</th>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>BAR</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BDI</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BEL</th>
      <td>141.0</td>
      <td>88.0</td>
      <td>161.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>BER</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BLR</th>
      <td>24.0</td>
      <td>8.0</td>
      <td>13.0</td>
      <td>29.0</td>
      <td>6.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>BOH</th>
      <td>5.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>132.0</td>
      <td>44.0</td>
      <td>86.0</td>
      <td>42.0</td>
      <td>15.0</td>
      <td>53.0</td>
    </tr>
    <tr>
      <th>BUL</th>
      <td>80.0</td>
      <td>40.0</td>
      <td>94.0</td>
      <td>56.0</td>
      <td>13.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>BWI</th>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>144.0</td>
      <td>122.0</td>
      <td>145.0</td>
      <td>83.0</td>
      <td>32.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>CHI</th>
      <td>21.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>CHN</th>
      <td>48.0</td>
      <td>95.0</td>
      <td>75.0</td>
      <td>145.0</td>
      <td>139.0</td>
      <td>177.0</td>
    </tr>
    <tr>
      <th>CIV</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CMR</th>
      <td>1.0</td>
      <td>18.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>COL</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CRC</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>CRO</th>
      <td>15.0</td>
      <td>31.0</td>
      <td>29.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>CUB</th>
      <td>76.0</td>
      <td>116.0</td>
      <td>107.0</td>
      <td>33.0</td>
      <td>44.0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>CZE</th>
      <td>10.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>DEN</th>
      <td>130.0</td>
      <td>95.0</td>
      <td>177.0</td>
      <td>22.0</td>
      <td>52.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>SRI</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>SUD</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SUI</th>
      <td>130.0</td>
      <td>71.0</td>
      <td>149.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SVK</th>
      <td>8.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>SWE</th>
      <td>284.0</td>
      <td>328.0</td>
      <td>320.0</td>
      <td>41.0</td>
      <td>19.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>SYR</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TAN</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TCH</th>
      <td>94.0</td>
      <td>60.0</td>
      <td>95.0</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>TGA</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>THA</th>
      <td>6.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>TJK</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TOG</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TPE</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>25.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>TRI</th>
      <td>11.0</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TUN</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TUR</th>
      <td>20.0</td>
      <td>36.0</td>
      <td>19.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>UAE</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>UGA</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>UKR</th>
      <td>38.0</td>
      <td>15.0</td>
      <td>22.0</td>
      <td>40.0</td>
      <td>17.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>URS</th>
      <td>419.0</td>
      <td>575.0</td>
      <td>482.0</td>
      <td>165.0</td>
      <td>263.0</td>
      <td>145.0</td>
    </tr>
    <tr>
      <th>URU</th>
      <td>30.0</td>
      <td>44.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>766.0</td>
      <td>1520.0</td>
      <td>834.0</td>
      <td>286.0</td>
      <td>568.0</td>
      <td>361.0</td>
    </tr>
    <tr>
      <th>UZB</th>
      <td>7.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>6.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>VIE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>YUG</th>
      <td>102.0</td>
      <td>125.0</td>
      <td>146.0</td>
      <td>16.0</td>
      <td>18.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>ZAM</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ZIM</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>ZZX</th>
      <td>8.0</td>
      <td>23.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>138 rows × 6 columns</p>
</div>




```python
genderGold = medals.pivot_table(index='NOC', columns=['Gender', 'Medal'], values = 'Athlete', aggfunc='count')
```

Here, the goal is the access the **Multi-Index** column and select only Gold column of the second level. This can be done as follows:


```python
genderGold.loc[:, (['Men', 'Women'], 'Gold')]

# or using slice(None) to specify all columns.
#genderGold.loc[:, (slice(None), 'Gold')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Gender</th>
      <th>Men</th>
      <th>Women</th>
    </tr>
    <tr>
      <th>Medal</th>
      <th>Gold</th>
      <th>Gold</th>
    </tr>
    <tr>
      <th>NOC</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AFG</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>AHO</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ALG</th>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>ANZ</th>
      <td>19.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ARG</th>
      <td>68.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ARM</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>AUS</th>
      <td>148.0</td>
      <td>145.0</td>
    </tr>
    <tr>
      <th>AUT</th>
      <td>17.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>AZE</th>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>BAH</th>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>BAR</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BDI</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BEL</th>
      <td>88.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>BER</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BLR</th>
      <td>8.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>BOH</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>44.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>BUL</th>
      <td>40.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>BWI</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>122.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>CHI</th>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CHN</th>
      <td>95.0</td>
      <td>139.0</td>
    </tr>
    <tr>
      <th>CIV</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CMR</th>
      <td>18.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>COL</th>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>CRC</th>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>CRO</th>
      <td>31.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CUB</th>
      <td>116.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>CZE</th>
      <td>6.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>DEN</th>
      <td>95.0</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>SRI</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SUD</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SUI</th>
      <td>71.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SVK</th>
      <td>8.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>SWE</th>
      <td>328.0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>SYR</th>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>TAN</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TCH</th>
      <td>60.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>TGA</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>THA</th>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>TJK</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TOG</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TPE</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>TRI</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TUN</th>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TUR</th>
      <td>36.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>UAE</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>UGA</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>UKR</th>
      <td>15.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>URS</th>
      <td>575.0</td>
      <td>263.0</td>
    </tr>
    <tr>
      <th>URU</th>
      <td>44.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>1520.0</td>
      <td>568.0</td>
    </tr>
    <tr>
      <th>UZB</th>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>VIE</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>YUG</th>
      <td>125.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>ZAM</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ZIM</th>
      <td>NaN</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>ZZX</th>
      <td>23.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>138 rows × 2 columns</p>
</div>




```python
# store this into a dataframe
genderGold = genderGold.loc[:, (['Men', 'Women'], 'Gold')]

# Replace NaNs with 0
genderGold.fillna(value=0, inplace=True)

genderGold.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Gender</th>
      <th>Men</th>
      <th>Women</th>
    </tr>
    <tr>
      <th>Medal</th>
      <th>Gold</th>
      <th>Gold</th>
    </tr>
    <tr>
      <th>NOC</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AFG</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>AHO</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ALG</th>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>ANZ</th>
      <td>19.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ARG</th>
      <td>68.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filter rows which have the desired criteria
# For the sake of simplicity, create a boolean Series
womenGold = genderGold.loc[:,(['Women'], 'Gold')]
```


```python
womenGold.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 138 entries, AFG to ZZX
    Data columns (total 1 columns):
    (Women, Gold)    138 non-null float64
    dtypes: float64(1)
    memory usage: 2.2+ KB



```python
# Filter rows which have the desired criteria
# For the sake of simplicity, create a boolean Series
# genderGold.loc[:, (['Women'], 'Gold')] >= genderGold.loc[:, (['Men'], 'Gold')]
```


```python
medals.pivot_table(index='NOC', columns='Medal', values= 'Athlete', aggfunc='count')
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
      <th>Medal</th>
      <th>Bronze</th>
      <th>Gold</th>
      <th>Silver</th>
    </tr>
    <tr>
      <th>NOC</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AFG</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>AHO</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ALG</th>
      <td>8.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>ANZ</th>
      <td>5.0</td>
      <td>20.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>ARG</th>
      <td>88.0</td>
      <td>68.0</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>ARM</th>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>AUS</th>
      <td>413.0</td>
      <td>293.0</td>
      <td>369.0</td>
    </tr>
    <tr>
      <th>AUT</th>
      <td>44.0</td>
      <td>21.0</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>AZE</th>
      <td>9.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>BAH</th>
      <td>5.0</td>
      <td>9.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>BAR</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BDI</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BEL</th>
      <td>150.0</td>
      <td>91.0</td>
      <td>167.0</td>
    </tr>
    <tr>
      <th>BER</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BLR</th>
      <td>53.0</td>
      <td>14.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>BOH</th>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>174.0</td>
      <td>59.0</td>
      <td>139.0</td>
    </tr>
    <tr>
      <th>BUL</th>
      <td>136.0</td>
      <td>53.0</td>
      <td>142.0</td>
    </tr>
    <tr>
      <th>BWI</th>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>227.0</td>
      <td>154.0</td>
      <td>211.0</td>
    </tr>
    <tr>
      <th>CHI</th>
      <td>21.0</td>
      <td>3.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>CHN</th>
      <td>193.0</td>
      <td>234.0</td>
      <td>252.0</td>
    </tr>
    <tr>
      <th>CIV</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>CMR</th>
      <td>1.0</td>
      <td>20.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>COL</th>
      <td>7.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>CRC</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>CRO</th>
      <td>18.0</td>
      <td>31.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>CUB</th>
      <td>109.0</td>
      <td>160.0</td>
      <td>126.0</td>
    </tr>
    <tr>
      <th>CZE</th>
      <td>13.0</td>
      <td>10.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>DEN</th>
      <td>152.0</td>
      <td>147.0</td>
      <td>192.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>SRI</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>SUD</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>SUI</th>
      <td>138.0</td>
      <td>73.0</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SVK</th>
      <td>8.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>SWE</th>
      <td>325.0</td>
      <td>347.0</td>
      <td>349.0</td>
    </tr>
    <tr>
      <th>SYR</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>TAN</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>TCH</th>
      <td>105.0</td>
      <td>80.0</td>
      <td>144.0</td>
    </tr>
    <tr>
      <th>TGA</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>THA</th>
      <td>10.0</td>
      <td>7.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>TJK</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>TOG</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TPE</th>
      <td>13.0</td>
      <td>2.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>TRI</th>
      <td>11.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>TUN</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>TUR</th>
      <td>22.0</td>
      <td>37.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>UAE</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>UGA</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>UKR</th>
      <td>78.0</td>
      <td>32.0</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>URS</th>
      <td>584.0</td>
      <td>838.0</td>
      <td>627.0</td>
    </tr>
    <tr>
      <th>URU</th>
      <td>30.0</td>
      <td>44.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>1052.0</td>
      <td>2088.0</td>
      <td>1195.0</td>
    </tr>
    <tr>
      <th>UZB</th>
      <td>8.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>8.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>VIE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>YUG</th>
      <td>118.0</td>
      <td>143.0</td>
      <td>174.0</td>
    </tr>
    <tr>
      <th>ZAM</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ZIM</th>
      <td>1.0</td>
      <td>18.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>ZZX</th>
      <td>10.0</td>
      <td>23.0</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
<p>138 rows × 3 columns</p>
</div>




```python
# Construct the pivot table: counted
counted = medals.pivot_table (index='NOC', columns='Medal', values='Athlete', aggfunc='count')

# Create the new column: counted['totals']
counted['totals'] = counted.sum(axis='columns')

# Sort counted by the 'totals' column
counted = counted.sort_values('totals', ascending=False)

# Print the top 15 rows of counted
print(counted.head(15))
```

    Medal  Bronze    Gold  Silver  totals
    NOC                                  
    USA    1052.0  2088.0  1195.0  4335.0
    URS     584.0   838.0   627.0  2049.0
    GBR     505.0   498.0   591.0  1594.0
    FRA     475.0   378.0   461.0  1314.0
    ITA     374.0   460.0   394.0  1228.0
    GER     454.0   407.0   350.0  1211.0
    AUS     413.0   293.0   369.0  1075.0
    HUN     345.0   400.0   308.0  1053.0
    SWE     325.0   347.0   349.0  1021.0
    GDR     225.0   329.0   271.0   825.0
    NED     320.0   212.0   250.0   782.0
    JPN     270.0   206.0   228.0   704.0
    CHN     193.0   234.0   252.0   679.0
    RUS     240.0   192.0   206.0   638.0
    ROU     282.0   155.0   187.0   624.0

