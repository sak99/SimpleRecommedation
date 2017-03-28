
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('F:/Data Analysis/PythonProgs/recommendation/ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('F:/Data Analysis/PythonProgs/recommendation/ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('F:/Data Analysis/PythonProgs/recommendation/ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')


# In[3]:

print users.shape
users.head


# In[4]:

print ratings.shape
print ratings.head


# In[5]:

print items.shape
print items.head


# In[6]:

items


# In[7]:

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('F:/Data Analysis/PythonProgs/recommendation/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('F:/Data Analysis/PythonProgs/recommendation/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_base.shape, ratings_test.shape


# In[8]:

import graphlab
train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)


# In[9]:

train_data


# In[10]:

popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')


# In[11]:

#Get recommendations for first 5 users and print them
#users = range(1,6) specifies user ID of first 5 users
#k=5 specifies top 5 recommendations to be given
popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)


# In[13]:

ratings_base.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20)
p


# In[14]:

#Train Model
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')

#Make Recommendations:
item_sim_recomm = item_sim_model.recommend(users=range(1,6),k=5)
item_sim_recomm.print_rows(num_rows=25)


# In[16]:

model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])


# In[ ]:




# In[ ]:



