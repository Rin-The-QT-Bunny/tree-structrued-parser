# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 14:32:40 2021

@author: doudou
"""

actions = ["add","subtract","multiply","divide","log","sqrt","factorial","gcd","lcm","power","max","min"
,"reminder","reminder","negate","inverse","round","floor","sine","cosine","tangent","radians_to_degree","degree_to_radians"
]


operators = ["root","get_name","query","get_index","change_state","Positive","Negative"]
ops = [1,2,3,4,5,6]

arged_ops = ["get_name","get_index","change_state"]
arith = ["add","substract","multiply","divide","log","sqrt","power"]
arguments = [0,1,2,3,4]

codes = ["change_state(get_index(get_name(query)),Positive)",
"change_state(get_index(get_name(query)),Negative)"]

tasks = [["what is the name of from the query", "get_name(query)"],
         ["give the name from the query","get_name(query)"],
         ["give me the name from the query","get_name(query)"],
         ["what is the name in the query","get_name(query)"],
         ["name of the query is","get_name(query)"],
         ["what is the index of name","get_index(get_name(query))"],
         ["what is the name index","get_index(get_name(query))"],
         ["give me the index of the name","get_index(get_name(query))"],
         ["give index of the name","get_index(get_name(query))"],
         ["what is the index of the name","get_index(get_name(query))"],
         ["change the name to negative","change_state(get_index(get_name(query)),Negative)"],
         ["set the name to negative","change_state(get_index(get_name(query)),Negative)"],
         ["name is the negative","change_state(get_index(get_name(query)),Negative)"],
         ["name is the positive","change_state(get_index(get_name(query)),Positive)"],
         ["set name to negative state","change_state(get_index(get_name(query)),Negative)"],
         ["set name to positive state","change_state(get_index(get_name(query)),Positive)"],
         ["name is negative","change_state(get_index(get_name(query)),Negative)"],
         ["name is positive","change_state(get_index(get_name(query)),Positive)"],
         ["name is in positive state","change_state(get_index(get_name(query)),Positive)"],
         ["name is in negative state","change_state(get_index(get_name(query)),Negative)"],
         ["name is negative state","change_state(get_index(get_name(query)),Negative)"],
         ["name is positive state","change_state(get_index(get_name(query)),Positive)"]
         ]

arg_dict = {
    "get_name":[1],
    "get_index":[2],
    "change_state":[3,4]
    }


texts= []

EPOCH = 1400
BATCH_SIZE = 1
"""
import pytesseract
from PIL import Image
text = pytesseract.image_to_string(Image.open('C:/Users/doudou/Pictures/likethis.jpg'),lang='chi_sim')  # chi_sim是简体中文训练包，如果想识别英文去掉lang选项即可
print(text)
"""