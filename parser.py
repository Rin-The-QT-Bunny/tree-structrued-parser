# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:09:03 2021

@author: Rin Sun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import time # record the time record
import matplotlib.pyplot as plt #draw the diagram

import numpy as np #linear algebra
import datetime
import random
from math import *
from dataloader import *
from config import *
import config


class Parser(nn.Module):
    def __init__(self,op_dim = 64, args_dim = 32,state_dim = 128, vocab = 3000):
        super(Parser, self).__init__()
        
        # create a list of vectors representing operators and words
        self.wordvecs = nn.Embedding(vocab, 64)
        self.op_vecs = nn.Embedding(vocab, op_dim)
        self.arg_vecs = nn.Embedding(vocab, args_dim)

        # Input of this model will be a set of embeddings of the operator
        
        # Recreate operators using pytorch
        self.state_dim = state_dim
        self.args_dim = args_dim
        self.op_dim = op_dim
        
        # Synthesis methods required
        self.log_p = 0
        self.count = 0
        self.program = ""
        
        # to create the probability analyzer unit
        self.pfc1 = nn.Linear(op_dim+state_dim,300)
        self.pfc2 = nn.Linear(300,200)
        self.pfca = nn.Linear(200,200)
        self.pfc3 = nn.Linear(200,1)
        # to create the repeater unit
        self.rfc1 = nn.Linear(args_dim+state_dim,200)
        self.rfc2 = nn.Linear(200,399)
        self.rfc3 = nn.Linear(399,state_dim)
        # Gated recurrential unit
        self.GRU = torch.nn.GRU(64,state_dim,1)
        self.namo = "Melian"
    
    def pdf_ops(self,opp,state):
        # shape of input variables are [num, embed_dim]
        # output will be the pdf tensor and the argmax position
        states = torch.broadcast_to(state,[len(opp),self.state_dim])

        r = torch.cat([opp,states],1)

        pdf_tensor = F.tanh(self.pfc1(r))
        pdf_tensor = F.softplus(self.pfc2(pdf_tensor))
        pdf_tensor = F.softplus(self.pfc3(pdf_tensor))
        #create pdf tensor
        pdf = pdf_tensor/torch.sum(pdf_tensor,0)
        
        #index of the maximum operator is
        index = np.argmax(pdf.detach().numpy())
        return pdf, index
    
    def repeat_vector(self,semantics,arg):
        # Create next semantics vector for the argument
        r = torch.cat([semantics,arg],-1)
        r = F.tanh(self.rfc1(r))
        r = F.softplus(self.rfc2(r))
        r = F.tanh(self.rfc3(r))
        
        return r
    
    def convert(self,x):
        # Start the parsing process
        self.log_p = 0
        self.program = ""
        def parse(s,arg,ops,ops_dict):
            # analogue goal-based policy in multi-goal RL
            # estimate the action to take when given the state (s) and the goal (arg)
            parse_state = self.repeat_vector(s,arg)
            # 1. create a state given the current state and the goal given
            pdf,index = self.pdf_ops(ops,parse_state)
            self.log_p  = self.log_p - torch.log(pdf[index])
            operator = ops_dict[index+1]
            self.program += str(operator)
            # 2. pass the semantics down to next level if required

            if operator in arged_ops:
                self.program +=  "("
                
                args = arg_dict[operator]
                args_paramed = self.arg_vecs(torch.tensor(args))

                for i in range(len(args)):
                    
                    parse(parse_state,torch.reshape(args_paramed[i],[1,self.args_dim]),ops,ops_dict)
                    if i != (len(args)-1):
                        self.program += ","
                self.program += ")"
    
        root = self.arg_vecs(torch.tensor([0]))
        ops = self.op_vecs(torch.tensor(config.ops))
        
        # TODO: dynamic library useage during the parsing
        parse(x,root,ops,operators)
        return self.program,self.log_p
    
    def program_prob(self,x,ops_sequence):
        # Start the parsing process
        self.log_p = 0
        self.count = 0
        self.program = ""
        def parse(s,arg,ops,ops_dict,ops_sequence):
            # analogue goal-based policy in multi-goal RL
            # estimate the action to take when given the state (s) and the goal (arg)
            parse_state = self.repeat_vector(s,arg)
            # 1. create a state given the current state and the goal given
            pdf,index = self.pdf_ops(ops,parse_state)
            index = ops_dict.index(ops_sequence[self.count])
            self.log_p  = self.log_p - torch.log(pdf[index-1])
            operator = ops_dict[index]
            self.program += str(operator)
            # 2. pass the semantics down to next level if required
            self.count += 1
            
            if operator in arged_ops:
                self.program +=  "("
                
                args = arg_dict[operator]
                args_paramed = self.arg_vecs(torch.tensor(args))

                for i in range(len(args)):
                    
                    parse(parse_state,torch.reshape(args_paramed[i],[1,self.args_dim]),ops,ops_dict,ops_sequence)
                    if i != (len(args)-1):
                        self.program += ","
                self.program += ")"
    
        root = self.arg_vecs(torch.tensor([0]))
        ops = self.op_vecs(torch.tensor(config.ops))
        
        # TODO: dynamic library useage during the parsing
        parse(x,root,ops,operators,ops_sequence)
        return self.program,self.log_p
    
    def save_model(self):
        dirc = "D:/AnQira/" + str(self.namo) + "/"
        torch.save(self, dirc+self.namo + '.pth')
        
    def load_model(self):
        dirc = "D:/AnQira/" + str(self.namo) + "/"
        try:
            self = torch.load(dirc+self.namo + '.pth')
        except:
            print("Failed")
            
    def run(self,x,seq = None):
        sequence = self.wordvecs(torch.tensor(x))
        
        results,hidden  = self.GRU(sequence)

        hidden = torch.reshape(hidden,[hidden.shape[1],128])
        semantics = torch.reshape(hidden[hidden.shape[0]-1],[1,128])

        if seq != None:
            program,loss  = self.program_prob(semantics,seq)
        else:
            program,loss = self.convert(semantics)
        return program, loss
   

ps = Parser() #Parser that coverts the sequence



Adam = torch.optim.Adam(ps.parameters(),0.0003) # An adam optimizer is initiated

for epoch in range(EPOCH):
    Loss = 0
    for i in range(len(tasks)):
        toks = data.convert_to_data(tasks[i][0])
        program, loss = ps.run(toks,Decompose(tasks[i][1]))
            
            
        Loss = Loss + loss
        #print(tasks[i][0],tasks[i][1])
    ps.zero_grad()
    Loss.backward()
    Adam.step()
    
    print("EPOCH:",epoch,"Loss: ",Loss.detach().numpy())


ps.save_model()
us = Parser()
us.load_model()
num = 0
for i in range(len(tasks)):
    toks = data.convert_to_data(tasks[i][0])
    program, loss = ps.run(toks,Decompose(tasks[i][1]))
    l = loss.detach().numpy()
    print("prob:",exp(-l))
    if (program == tasks[i][1]):
        num = num + 1
    else:
        print(tasks[i][0])
        print(program,tasks[i][1])
        
def convert_to_program(x,tar):
    toks = data.convert_to_data(x)
    program, loss = tar.run(toks,Decompose(tasks[i][1]))
    return program,exp(-loss)
        