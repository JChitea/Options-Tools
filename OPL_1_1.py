# Make the necessary imports
import tkinter as tk
from tkinter import filedialog
import openpyxl
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np
import yfinance as yf
import math

# Define the class structure for the GUI

class OPL_GUI(object):
     """Class wrapper for the GUI"""

     def __init__(self):
        """Constrcutor"""
        self.holdings = dict()

        # Start by establishing the main window
        self.root = tk.Tk()
        self.root.geometry('1500x700')
        self.root.title('Options P&L Analysis')
        self.root.columnconfigure(0, weight=4)
        self.root.columnconfigure(1, weight=4)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=3)
        self.root.rowconfigure(2, weight=1)


        # Okay, now we'll begin to build out the left side

        # We'll start by adding the File Adding Feature
        # Make a label
        self.fileSelectFrame = tk.Frame(self.root)
        self.fileSelectFrame.grid(row=0,column=0, sticky=tk.NS+tk.EW)
        self.fileLabel = tk.Label(self.fileSelectFrame,text='Add Options Chain File', bg='#B8CFEB')
        self.fileLabel.pack(fill='x')
        self.fileBut = tk.Button(self.fileSelectFrame, text='Retreive Contracts', command=self.retrieve_options)
        self.fileBut.pack(fill='x')

        
        # We need to enter the selected options fromt eh selected file into the 
        # "available options" listbox, which does not yet exist. So we make it
        self.availFrame = tk.Frame(self.root)
        self.availFrame.grid(row=1,column=0, sticky=tk.NS+tk.EW)
        self.availLabel = tk.Label(self.availFrame, text='Available Options', bg='#B8CFEB')
        self.availLabel.pack(fill='x')
        self.availOpt = tk.Listbox(self.availFrame)
        self.availOpt.pack(side='left',fill='both', expand=True)
        self.availScroll = tk.Scrollbar(self.availFrame)
        self.availScroll.pack(side='right', fill='both')
        self.availOpt.config(yscrollcommand=self.availScroll.set)
        self.availScroll.config(command=self.availOpt.yview)
        # We aren't done with this section just yet. We need to include and "add" button
        # So that the user can add that contract to the current portfolio
        self.addbutton = tk.Button(self.availFrame, text='Add', command=self.add_to_holdings)
        self.addbutton.pack(side='bottom', fill='both', expand=True)

        # Now we'll get to building out the current holdings section
        self.curFrame = tk.Frame(self.root)
        self.curFrame.grid(row=2,column=0,sticky=tk.NS+tk.EW)
        self.curLabel = tk.Label(self.curFrame, text='Current Holdings', bg='#B8CFEB')
        self.curLabel.pack(fill='x')
        self.curOpt = tk.Listbox(self.curFrame)
        self.curOpt.pack(side='left',fill='both', expand=True)
        self.curScroll = tk.Scrollbar(self.curFrame)
        self.curScroll.pack(side='right', fill='both')
        self.curOpt.config(yscrollcommand=self.curScroll.set)
        self.curScroll.config(command=self.curOpt.yview)
        # We aren't done with this section just yet. We need to include and "add" button
        # So that the user can add that contract to the current portfolio
        self.delButton = tk.Button(self.curFrame, text='Delete', command=self.remove_holdings)
        self.delButton.pack(side='bottom', fill='both', expand=True)

        # END LEFT SIDE OF GUI DEVELOPMENT, BEGIN RIGHT SIDE GUI DEVELOPMENT

        # On the right side, we'll 
        self.plotFrame = tk.Frame(self.root)
        self.plotFrame.grid(row=0,column=1, rowspan=3, sticky=tk.NS+tk.EW)
        self.PLG = Figure(figsize=(8,4), dpi=85)
        self.PLG_Canvas = FigureCanvasTkAgg(self.PLG, self.plotFrame)
        self.PLG_Canvas.draw()
        self.PLG_axes = self.PLG.add_subplot(1,1,1)
        self.PLG_axes.grid(True)
        self.PLG_axes.set_title('Profit and Loss Analysis of Current Options')
        self.PLG_axes.set_ylabel('Profit/Loss($)')
        self.PLG_axes.set_xlabel('Underlying Price')
        NavigationToolbar2Tk(self.PLG_Canvas, self.plotFrame)
        
        self.PLG_Canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


        
        self.root.mainloop()

     def retrieve_options(self):
         self.new_window = tk.Toplevel()
         self.new_window.title("Ticker Options Chain Selection")
         self.new_window.geometry("500x500")
         self.new_window.rowconfigure(0,weight=1)
         self.new_window.rowconfigure(1,weight=3)
         self.new_window.rowconfigure(2,weight=1)
         self.new_window.columnconfigure(0,weight=1)

         self.tckr_frame = tk.Frame(self.new_window)
         self.tckr_frame.grid(row=0,column=0,columnspan=2,sticky=tk.NS+tk.EW)
         self.search_label = tk.Label(self.tckr_frame, text='Input Ticker', bg='#B8CFEB')
         self.search_label.pack(fill='x')
         self.tckr_entry = tk.Entry(self.tckr_frame)
         self.tckr_entry.pack(fill='x',expand=True)
         self.submit_button = tk.Button(self.tckr_frame, text='Submit', command=self.get_expirys)
         self.submit_button.pack(fill='x', side='left',pady=10)

         self.expr_frame = tk.Frame(self.new_window)
         self.expr_frame.grid(row=1,column=0,columnspan=2,sticky=tk.NS+tk.EW)
         self.exprs_box_label = tk.Label(self.expr_frame, text='Available Expiries', bg='#B8CFEB')
         self.exprs_box_label.pack(fill='x')
         self.exprs_box = tk.Listbox(self.expr_frame, selectmode='multiple')
         self.exprs_box.pack(side='left',fill='both',expand=True)
         self.exprs_scroll = tk.Scrollbar(self.expr_frame)
         self.exprs_scroll.pack(side='right',fill='both')
         self.exprs_box.config(yscrollcommand=self.exprs_scroll.set)
         self.exprs_scroll.config(command=self.exprs_box.yview)

         self.tri_butt_frame = tk.Frame(self.new_window)
         self.tri_butt_frame.grid(row=2,column=0,columnspan=2,sticky=tk.NS+tk.EW)
         self.put_button = tk.Button(self.tri_butt_frame, text='Puts', command=self.add_put_chain)
         self.put_button.pack(fill='x')
         self.call_button = tk.Button(self.tri_butt_frame, text='Calls', command=self.add_call_chain)
         self.call_button.pack(fill='x')
         self.all_button = tk.Button(self.tri_butt_frame, text='All', command=self.add_both_chain)
         self.all_button.pack(fill='x')

    
     def get_expirys(self):
         # Start by getting the entered ticker from the entry 
         # field
         tckr = yf.Ticker(self.tckr_entry.get().strip().upper())
         exprs = tckr.options

         # Check if there are no available options
         if len(exprs) == 0:
             no_opt_message = tk.Toplevel(self.new_window)
             no_opt_message.title('Error Message')
             no_opt_message.geometry('400x100')
             tk.Label(no_opt_message, text='No options found with this ticker symbol',bg='#B8CFEB').pack(expand=True)
         else:
             # We need to delete all of the current expiry options currently in the box, if they exist
             self.exprs_box.delete(0,tk.END)
             # And the next thing we need to do is add all of the expiry dates
             for date in exprs:
                 self.exprs_box.insert(tk.END,date)

     def add_both_chain(self):
         tckr = yf.Ticker(self.tckr_entry.get().strip().upper())
         exprs = tckr.options

         # Check if there are no available options
         if len(exprs) == 0:
             no_opt_message = tk.Toplevel(self.new_window)
             no_opt_message.title('Error Message')
             no_opt_message.geometry('400x100')
             tk.Label(no_opt_message, text='No options found with this ticker symbol',bg='#B8CFEB').pack(expand=True)
         else:
             # We need to know which expirys are selected 
             picked_dates = [self.exprs_box.get(i) for i in self.exprs_box.curselection()]
             # Now, we'll need to clear out the current avaibale contracts list before addint
             # the new ones in it
             self.availOpt.delete(0, tk.END)
             self.availOpt.insert(0, 'Contract Type  |  Expiration  |  Contract ID  |  Strike  |  Last Price |  ITM  |  Implied Volatility')
             for date in picked_dates:
                 date_contracts = tckr.option_chain(date)
                 date_calls = date_contracts.calls
                 date_calls = date_calls.iloc[0:,[0,2,3,10,11]]
                 for i in range(0, date_calls.shape[0]):
                     self.availOpt.insert(tk.END, f'CALL  |  {date}  |  {date_calls.iloc[i]["contractSymbol"]}  |  {date_calls.iloc[i]["strike"]}  |  {date_calls.iloc[i]["lastPrice"]}  |  {date_calls.iloc[i]["inTheMoney"]}  |  {date_calls.iloc[i]["impliedVolatility"]}')
                 date_puts = date_contracts.puts
                 date_puts = date_puts.iloc[0:,[0,2,3,10,11]]
                 for i in range(0, date_puts.shape[0]):
                     self.availOpt.insert(tk.END, f'PUT   |  {date}  |  {date_puts.iloc[i]["contractSymbol"]}  |  {date_puts.iloc[i]["strike"]}  |  {date_puts.iloc[i]["lastPrice"]}  |  {date_puts.iloc[i]["inTheMoney"]}  |  {date_puts.iloc[i]["impliedVolatility"]}')

     def add_put_chain(self):
         tckr = yf.Ticker(self.tckr_entry.get().strip().upper())
         exprs = tckr.options

         # Check if there are no available options
         if len(exprs) == 0:
             no_opt_message = tk.Toplevel(self.new_window)
             no_opt_message.title('Error Message')
             no_opt_message.geometry('400x100')
             tk.Label(no_opt_message, text='No options found with this ticker symbol',bg='#B8CFEB').pack(expand=True)
         else:
             # We need to know which expirys are selected 
             picked_dates = [self.exprs_box.get(i) for i in self.exprs_box.curselection()]
             # Now, we'll need to clear out the current avaibale contracts list before addint
             # the new ones in it
             self.availOpt.delete(0, tk.END)
             self.availOpt.insert(0, 'Contract Type  |  Expiration  |  Contract ID  |  Strike  |  Last Price |  ITM  |  Implied Volatility')
             for date in picked_dates:
                 date_contracts = tckr.option_chain(date)
                 date_puts = date_contracts.puts
                 date_puts = date_puts.iloc[0:,[0,2,3,10,11]]
                 for i in range(0, date_puts.shape[0]):
                     self.availOpt.insert(tk.END, f'PUT   |  {date}  |  {date_puts.iloc[i]["contractSymbol"]}  |  {date_puts.iloc[i]["strike"]}  |  {date_puts.iloc[i]["lastPrice"]}  |  {date_puts.iloc[i]["inTheMoney"]}  |  {date_puts.iloc[i]["impliedVolatility"]}')
          
     def add_call_chain(self):
         tckr = yf.Ticker(self.tckr_entry.get().strip().upper())
         exprs = tckr.options

         # Check if there are no available options
         if len(exprs) == 0:
             no_opt_message = tk.Toplevel(self.new_window)
             no_opt_message.title('Error Message')
             no_opt_message.geometry('400x100')
             tk.Label(no_opt_message, text='No options found with this ticker symbol',bg='#B8CFEB').pack(expand=True)
         else:
             # We need to know which expirys are selected 
             picked_dates = [self.exprs_box.get(i) for i in self.exprs_box.curselection()]
             # Now, we'll need to clear out the current avaibale contracts list before addint
             # the new ones in it
             self.availOpt.delete(0, tk.END)
             self.availOpt.insert(0, 'Contract Type  |  Expiration  |  Contract ID  |  Strike  |  Last Price |  ITM  |  Implied Volatility')
             for date in picked_dates:
                 date_contracts = tckr.option_chain(date)
                 date_calls = date_contracts.calls
                 date_calls = date_calls.iloc[0:,[0,2,3,10,11]]
                 for i in range(0, date_calls.shape[0]):
                     self.availOpt.insert(tk.END, f'CALL  |  {date}  |  {date_calls.iloc[i]["contractSymbol"]}  |  {date_calls.iloc[i]["strike"]}  |  {date_calls.iloc[i]["lastPrice"]}  |  {date_calls.iloc[i]["inTheMoney"]}  |  {date_calls.iloc[i]["impliedVolatility"]}')

        
     def add_to_holdings(self):
         # We start by getting the currently selected option
         option = self.availOpt.get(self.availOpt.curselection())
         add_str = option.split(sep='|')
         cur_label = self.availLabel.cget("text")
         if add_str[2].strip() not in self.holdings.keys():
            if 'PUT' in option:
                 opt_type = 'PUT'
            else:
                 opt_type = 'CALL'
                 
            self.holdings.update({add_str[2].strip():{'strike': float(add_str[3].strip()), 'cost': float(add_str[4].strip()), 'N': 1, 'type': opt_type, 'expiry': add_str[1].strip()}})
            # We need to insert the new holding into the listbox
            insert_str = self.format_holding(add_str[2].strip())
            self.curOpt.insert(tk.END, insert_str)
         else:
            self.holdings[add_str[2].strip()]['N'] += 1
            # We need to update the current entry now, how many options are entered right now?
            all_options = self.curOpt.get(0,tk.END)
            for option in all_options:
                if add_str[2].strip() in option:
                    # Get the new formatted string
                    alt_str = self.format_holding(add_str[2].strip())
                    place_idx = all_options.index(option)
                    if place_idx == len(all_options)-1:
                        self.curOpt.delete(tk.END)
                        self.curOpt.insert(tk.END, alt_str)
                    else:
                        self.curOpt.insert(place_idx+1, alt_str)
                        self.curOpt.delete(place_idx)
         # Now we need to update the graph
         X,Y = self.compute_PL()
         self.plot_PL(X,Y)
         

     def plot_PL(self, X, Y):
         self.PLG_axes.clear()
         self.PLG_axes.plot(X,Y, color='black')
         self.PLG_axes.grid(True)
         self.PLG_axes.set_title('Profit and Loss Analysis of Current Options', fontdict={'size': 18})
         self.PLG_axes.set_ylabel('Profit/Loss($)', fontdict={'size': 14})
         self.PLG_axes.set_xlabel('Underlying Price', fontdict={'size': 14})
         # Find where the options portfolio is in the money
         mask_1 = np.where(Y>0)
         if len(mask_1[0]) != 0:
            Y_temp = np.zeros(len(mask_1[0]))
            self.PLG_axes.fill_between(X[mask_1[0]], Y[mask_1[0]], Y_temp, color='green', alpha=0.5)
         mask_2 = np.where(Y<0)
         if len(mask_2[0]) != 0:
            Y_temp = np.zeros(len(mask_2[0]))
            self.PLG_axes.fill_between(X[mask_2[0]], Y[mask_2[0]], Y_temp, color='red', alpha=0.5)
         self.PLG_axes.set_xticks(np.arange(0,np.max(X),math.floor(math.sqrt(np.max(X)))))
         self.PLG_axes.set_yticks(np.arange(np.min(Y)-25,np.max(Y)+25,math.floor(math.sqrt(np.max(Y)-np.min(Y)))))
         
         self.PLG_Canvas.draw()
     
     def compute_PL(self):
         # start by getting the mean of all the strike prices
         strikes = [self.holdings[entry]['strike'] for entry in self.holdings.keys()]
         # Get the min of the strike values 
         min_x = min(strikes)
         # Get the max of the strike values
         max_x = max(strikes)
         # Instantiate the domain over which the profit and loss function is 
         # defined
         X = np.arange(0,max_x+(min_x),0.01)
         # Now the output variable will be strike-dependant, so we iterate over the 
         # strike prcies
         Y = np.zeros(len(X))
         for key in self.holdings.keys():
             # obtain the strike
             strike  = self.holdings[key]['strike']
             # Obtain the cost, this will be a vertical 
             # displacement of the P and L function
             cost = self.holdings[key]['cost']*self.holdings[key]['N']
             # And finally obtain the type of option we are dealing with
             type = self.holdings[key]['type']
             if type == 'PUT':
                 mask_Y = strike >= X
                 Y[mask_Y] = Y[mask_Y]+((strike-X[mask_Y])*self.holdings[key]['N'])
             else:
                 mask_Y = strike <= X
                 Y[mask_Y] = Y[mask_Y]+((X[mask_Y]-strike)*self.holdings[key]['N'])
             Y = Y-cost
         return X,Y*100
     
     def remove_holdings(self):
          option = self.curOpt.get(self.curOpt.curselection())
          del_str = option.split(sep=',')

          # First we need to update the dictionary
          cur_num_held = self.holdings[del_str[0]]['N']
          if cur_num_held == 1:
              self.holdings.pop(del_str[0])
              all_options = self.curOpt.get(0,tk.END)
              for option in all_options:
                  if del_str[0] in option:
                    place_idx = all_options.index(option)
                    self.curOpt.delete(place_idx)
          else:
              self.holdings[del_str[0]]['N'] -= 1
              all_options = self.curOpt.get(0,tk.END)
              for option in all_options:
                  if del_str[0] in option:
                    # Get the new formatted string
                    alt_str = self.format_holding(del_str[0])
                    place_idx = all_options.index(option)
                    if place_idx == len(all_options)-1:
                        self.curOpt.delete(tk.END)
                        self.curOpt.insert(tk.END, alt_str)
                    else:
                        self.curOpt.insert(place_idx+1, alt_str)
                        self.curOpt.delete(place_idx)
          # Now we need to update the graph
          X,Y = self.compute_PL()
          self.plot_PL(X,Y)

          
     
     def format_holding(self, order_id):
         # Get the option in question
         ID = order_id
         strike = self.holdings[ID]['strike']
         cost = self.holdings[ID]['cost']
         N = self.holdings[ID]['N']
         Type = self.holdings[ID]['type']
         Expiry = self.holdings[ID]['expiry']
         add_str = ID+', '+str(strike)+', '+str(cost*N)+', '+Type+', '+str(N)+', '+Expiry
         return add_str
              

     def get_files(self):
          file_path = filedialog.askopenfilename(title='Select Options Chains Workbook')
          if len(file_path) != 0:
               # First we need to empty the previous list of options 
               self.availOpt.delete(0, tk.END)
                # Here we need to add every Options contract avialble to the first list box
                # We'll use opepyxl for this
               wb = openpyxl.load_workbook(file_path)
               ws = wb.active
               for row in ws.iter_rows(min_row=1):
                   add_str = str(row[0].value)+', '+str(row[2].value)+', '+str(row[3].value)+', '+str(row[10].value)
                   self.availOpt.insert(tk.END, add_str)
                # Update the label of the box to reflect what we're selecting
               ochain_TCKR = file_path.split(sep='/')[-1].split(sep='_')[0]
               ochain_Type = file_path.split(sep='/')[-1].split(sep='_')[1]
               if ochain_Type == 'P':
                    ochain_Type ='Puts'
               else:
                    ochain_Type ='Calls'
               ochain_FDate = file_path.split(sep='/')[-1].split(sep='_')[2]
               ochain_FDate = ochain_FDate[0:2]+'-'+ochain_FDate[2:4]+'-'+ochain_FDate[4:8]
                
               self.availLabel.configure(text=ochain_TCKR+' '+ochain_Type+' '+ochain_FDate)
          else:
               pass
          


OPL_GUI()




