import streamlit as st
import pandas as pd
import numpy as np


st.title('DL MACHINE LEARNING ')
st.write('HTS , Vietnamese translation - Classifier')

st.write('Accuracy: HTS 95%; Vietnamese 89%')


# # @st.cache
# # def load_data(nrows):
# #     data = pd.read_csv(DATA_URL, nrows=nrows)
# #     lowercase = lambda x: str(x).lower()
# #     data.rename(lowercase, axis='columns', inplace=True)
# #     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
# #     return data

# data_load_state = st.text('Loading data...')
# # data = load_data(100)

# data_load_state.text('Loading data... done!')

# # if st.checkbox('Show raw data'):
# #     st.subheader('Raw data')
# #     st.write(data)

# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)

# # Some number in the range 0-23
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)


import pickle
import pandas as pd
import numpy as np
import copy
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
import sqlalchemy
import pyodbc
    

class datalogic_hscode_vi:

    def __int__(self, data):
        _=datalogic_hscode_vi() # set obj inside to quick call sstatic method
        import_all_lib()
        self.data=data
        self.finaldata=[]
    # attribute ===================================================================
    
    pn_list=[]
    des_list=[]
    des_list_sample=([
    'ADDENDUM,REG,MGL3400/3500',
    'ASSY MODULE FIREFLY DE2101-DL GS',
    'OPTICAL CHAMBER SUBASSY PUL2 STD MIC',
    'LOADCELL,30KG,SHEKEL,C72D',
    'LOCKING SYST BASE ASSY LIGHT, CRADLE MIC',
    'FAB, FE4629, CUMULUS ILLUMINATION 100',
    'VF - 233 SCREW M1,4X0,3 L=4 TORX T3', 
    'CVL-1057 RS232 OUTPUT CABLE TC1200',
    'IC STR91XF 16/32-BIT FLASH MCU LQFP128',
      'CVL-1120 OUTPUT CABLE TC1200 SB4507'
    ])
    
    pn_list_sample=([
    'ADDENDUM,REG,MGL3400/3500',
    'ASSY MODULE FIREFLY DE2101-DL GS',
    'OPTICAL CHAMBER SUBASSY PUL2 STD MIC',
    'LOADCELL,30KG,SHEKEL,C72D',
    'LOCKING SYST BASE ASSY LIGHT, CRADLE MIC',
    'FAB, FE4629, CUMULUS ILLUMINATION 100',
    'VF - 233 SCREW M1,4X0,3 L=4 TORX T3', 
    'CVL-1057 RS232 OUTPUT CABLE TC1200',
    'IC STR91XF 16/32-BIT FLASH MCU LQFP128',
      'CVL-1120 OUTPUT CABLE TC1200 SB4507'
    ])
    finaldata=[]
    # self/instance method ===================================================================
    def import_all_lib(self):
        print('lib import done')
        
    def check_len_df(self):
        print("pn:" , len(self.pn_list))
        print("pn:" , len(self.des_list))
    
    
    def set_data_pn(self,data):
        self.pn_list=data
    def set_data_des(self,data):
        self.des_list=data
    
    def pn_dichviet(self):
        pass
        
    def pn_hscode(self):
        pass
        
    def pn_dichviet_hscode(self):
#         print('pn',self.pn_list)
        mydata=self.saplinkinfo_multi(self.pn_list)
        
#         print('eng from my data', mydata)
        return self.predict_data_vi_hs_from_pn(self.pn_list,mydata)

    def des_dichviet_hscode(self):
#         print('pn',self.pn_list)
        mydata=self.des_list
        
#         print('eng from my data', mydata)
        return self.predict_data_vi_hs_from_pn(self.des_list,mydata)

    def des_dichviet(self):
#         self.check_len_df()
        self.predict_from_description('vi',self.des_list)
        
    def des_hscode(self):
#         self.check_len_df()
        self.predict_from_description('hs',self.des_list)
        

        

    
    # class/cls method===================================================================

    
    # statis method ===================================================================

    
    @staticmethod
    def predict_from_description(target, datainput):
        filename = "VI_CLF.model" if (target=="" or target=="vi") else "HS_CLF.model" # defalut VI
#         print(filename)
        data = pd.Series(datainput)

        loaded_model = pickle.load(open(filename, 'rb'))
        predicted = loaded_model.predict((data))

        for doc, category in zip(data, predicted):
            print('%r => %s' % (doc, category))
        return predicted
    @staticmethod
    def goVector(X_text): # no need use
        tv = TfidfVectorizer()
        tv_X_train = tv.fit_transform(X_text)
        return tv_X_train
    
    @staticmethod
    def predict_data_vi_hs_from_pn(datainput, eng_list):
#         print('len là ',len(datainput))
#         print(self.data)
#         print(type(self.data)) # class dataFrame
        
#         material = (self.data.values)
        material = (datainput)
#         print('material',(material))
#         eng_list=saplinkinfo_multi(material) # return list description
    #     print(data_input)
    
        data = (eng_list)
#         print('data pn là:', material)
#         print('eng_list saplink là: ', eng_list)
#          -----------------
#version 1
#     list_model=["VI_CLF.model","HS_CLF.model"]
#     list_predict=[]
#     for item in range(len(list_model)):
# #         print(list_model[item])
#         model_name=list_model[item]

#         loaded_model = pickle.load(open(model_name, 'rb'))
#         predicted = loaded_model.predict((data))
#         list_predict.append(predicted)

#          -----------------
#version 2

        list_model=["VI_CLF.model","HS_CLF.model"]
        list_predict=[]
    #         for item in range(len(list_model)):
    #         print(list_model[item])
    #             model_name=list_model[item]

        loaded_model = pickle.load(open(list_model[0], 'rb'))
        predicted = loaded_model.predict(data)
        list_predict.append(predicted)
        loaded_model = pickle.load(open(list_model[1], 'rb'))

        #khangvan
        newdf= pd.DataFrame(dict(s1 = data, s2 = predicted)).reset_index()

        newdf["newdata"]=newdf['s1']+' '+newdf['s2']
        newdata =newdf["newdata"]

    #     for doc, category in zip(data,vi_df ):
    #         newdf4hs=data+" "+vi_df
    #         print(newdf4hs)
    #         newdata.append(newdf4hs)
        predicted = loaded_model.predict((newdata))
        list_predict.append(predicted)

            
#         ----------   
        main=[]
        # Vi
        for doc, category in zip(data, list_predict[0]):
    #         print('%r => %s' % (doc, category))
            v='%r => %s' % (doc, category)
    #         print(type(v))
            main.append('%r => %s' % (doc, category))
        #HS
        for doc, category in zip(main, list_predict[1]):
            print('%r => %s' % ((doc), category))

        def check_score_df(mytest):
            import pickle
            mydf=copy.copy(mytest)
            vi_df = pickle.load(open('VI_CLF_report.df', 'rb'))
            hs_df = pickle.load(open('HS_CLF_report.df', 'rb'))
            vi_df["vi"]=vi_df.index
            hs_df["hs"]=hs_df.index
            mydf =mydf.merge(vi_df, left_on='vi', right_on='vi')
            # print(mydf.columns)
            mydf.drop(columns=['precision','recall','support'],inplace=True)
            mydf.rename(columns={"f1-score": "vi_f1_score"}, inplace=True)

            mydf =mydf.merge(hs_df, left_on='hs', right_on='hs')
            mydf.drop(columns=['precision','recall','support'],inplace=True)
            mydf.rename(columns={"f1-score": "hs_f1_score"}, inplace=True)

            def mynote(row):
                vi_rate=row['vi_f1_score']
                hs_rate=row['hs_f1_score']
                cmt1= 'vi check' if vi_rate <0.5 else ''
                cmt2= 'hs check' if hs_rate <0.5 else ''
                return cmt1+ ' ' +cmt2
            mydf['mynote']=mydf.apply(mynote, axis=1)

            return mydf

        vi=list_predict[0]
        hs=list_predict[1]
        en= eng_list
        final_df= pd.DataFrame(vi,columns=["vi"])
        final_df['hs']=hs
        final_df['en']=en
        final_df['material_input']=material

        cols=['material_input','en', 'vi','hs']
        hidata=final_df[cols]
        
        returndata=check_score_df(hidata) #vi, hs, en
        import datetime as dt     
        date = dt.date.today()
        returndata['loaddate']= pd.to_datetime('today')
#         print(returndata)
        return returndata

    ## work now<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    @staticmethod
    def saplinkinfo_multi(material):
        
        import xml.etree.ElementTree as ET
        from urllib.request import urlopen
        from xml.etree.ElementTree import parse
        import time 
        import urllib.parse
        import asyncio
        start=time.time()
        
        def saplinkinfo_local(material='770112003'):
            import xml.etree.ElementTree as ET
            from urllib.request import urlopen
            from xml.etree.ElementTree import parse
            import time 
            import urllib.parse
            start=time.time()
        #     material='770112003'
            # tree = ET.parse('saplink.xml')
            tag=fr'''
            <Z_DMS_MATERIAL_INFO>
            <LANG>E</LANG>
            <WERKS></WERKS>
            <VALIDON>03/23/2015</VALIDON>
            <MATNR>{material}</MATNR>
            </Z_DMS_MATERIAL_INFO>
            '''
            encodetag=urllib.parse.quote_plus(tag)

            query='http://home/saplink/PRD/default.asp?XDoc='

            url =query+encodetag
        #     print(url)

            var_url = urlopen(url)
            tree = parse(var_url)

            # tree = ET.parse(url)
            root = tree.getroot()
            root.tag
            # for child in root:
            #     print(child.tag, child.attrib)

            # [elem.tag for elem in root.iter()]
            list_des=[]
            material_description=""
            for description in root.iter('DESCRIPTION'):
                material_description=description.text+ material[0:3]
        #         print(description.text)
                list_des.append(material_description)
        #     print("run in seconds: %s" % str(time.time()-start))
            return list_des, material
        
    #     material='770112003'
        # tree = ET.parse('saplink.xml')
        list_name=[]
        for item in range(len(material)):
#             print(material[item])
            eng_item= (saplinkinfo_local(material[item]))[0]
#             print(eng_item)
            list_name.append(eng_item[0])
        eng_list=list_name
        return eng_list
    # saplinkinfo_multi(['GD4130-BK','287021700'])
    
    #instnace method
    def get_pn_server(self):
        self.pn_list= (self.read_server_request())['Material']
#         print(self.pn_list)
        return list(self.pn_list)

    def pn_server_dichviet_hscode(self):
        self.get_pn_server()
        df=self.pn_dichviet_hscode()
        self.save_temp_server(df)
        print('done save file at server temp_hts_vi')
    def pn_new_dichviet_hscode(self):
#         self.get_pn_server()
        df=self.pn_dichviet_hscode()
        self.save_temp_server(df)
        print('done save file at server temp_hts_vi')

    @staticmethod
    def update_server_master():
        engine = sqlalchemy.create_engine("mssql+pymssql://reports:reports@vnmacsdb:1433/ACS EE")
        myquery='''
        insert into HTS_VI ( material_input, en, vi, hs, vi_f1_score, hs_f1_score, mynote, loaddate) 
        select * from temp_hts_vi where material_input not in (
        select material_input from  HTS_VI where loaddate >= DATEADD(minute,-1,getdate())
        
        )
        '''
        engine.execute(myquery)
        
        
        
        print('update complete, then show top 10 record from server')
        
        sql_view="select top 10 * from HTS_VI order by loaddate desc"
        df = pd.read_sql(sql_view, engine)

        return df

    

    @staticmethod
    def read_server_request():
        engine = sqlalchemy.create_engine("mssql+pymssql://reports:reports@vnmacsdb:1433/ACS EE")
        df = pd.read_sql('select top 100 Material  from hts_vi where DescriptionVI is null or HTSCode is null', engine)
#         df.head()
#         self.data=list(df.Material)
        return df
    @staticmethod
    def save_temp_server(newdf):
        engine = sqlalchemy.create_engine("mssql+pymssql://reports:reports@vnmacsdb:1433/ACS EE")
        newdf.to_sql(name="temp_hts_vi", con=engine, if_exists='replace', index=False,
                dtype={'input_data': sqlalchemy.String(),
                       'en': sqlalchemy.Text(), 
                        'vi': sqlalchemy.types.NVARCHAR(length=100), 
                       'hs': sqlalchemy.String() ,
                        'loaddate': sqlalchemy.DateTime(), 
    #                    'Amount LC': sqlalchemy.types.Float(precision=3, asdecimal=True), 
    #                    'Quantity': sqlalchemy.types.Numeric(),
                       })
        # engine = create_engine('mssql+pymssql://reports:reports@vnmacsdb:1433/ACS EE')
        df = pd.read_sql('select * from temp_hts_vi ', engine)
#         df.head()
        return df

    @staticmethod
    def check_bieuthue(code):
        # dl.update_result_to_server()
        my_report_name ='bieuthue2019'
        df = pickle.load(open(my_report_name, 'rb'))
        
        print(df.loc[df['hscode']==code])



#-----------------------------------------------------------   






import sys
caunoi="NHAP DATA"

title = st.text_area('{0}'.format(caunoi), '')
mylist=title.split('\n')

process=""

mylist=title.split('\n')
#send list to have list complate
while '' in mylist:
    mylist.remove('')
#st.write('My list',mylist)
st.write('Count qty',len(mylist))
    
if st.button('START HTSCODE VI'):
    
    process="pn"
   
    
elif st.button('START HTSCODE VI FROM DES'):
    
    process="des"
    
    


#d=st.date_input("ngay nhap data")
#st.write('Your birthday is:', d)

if(process=='pn'):

    
    dl=datalogic_hscode_vi()
    dl.set_data_pn(mylist)
    newdata=dl.pn_dichviet_hscode()
    dl.save_temp_server(newdata)
    dl.update_server_master()
    st.write(newdata)
    st.write('done')
    
elif(process=='des'):
    
    dl=datalogic_hscode_vi()
    dl.set_data_des(mylist)
    newdata=dl.des_dichviet_hscode()
    #dl.save_temp_server(newdata)
    #dl.update_server_master()
    st.write(newdata)
    st.write('done')

st.graphviz_chart('''
    digraph {
        material -> saplink
        saplink -> material_master_data
        material_master_data -> predict_vietnamese
        predict_vietnamese -> map_to_english
        map_to_english -> predict_HSTcode
        predict_HSTcode -> save_server
        save_server -> complete
       
    }
''')