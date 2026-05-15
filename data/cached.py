"""
data/cached.py — Streamlit-cache-lag.
Al @st.cache_data samlet ét sted. data/fetch.py er ren Python.
Skift til et andet framework: erstat kun denne fil.
                                         
                      
                                                                           
"""
         

                  
                   
import streamlit as st
from data.fetch import (
    fetch_price_history as _fetch_price_history,
    fetch_price_history_intraday as _fetch_price_history_intraday,
    fetch_live_quotes as _fetch_live_quotes,
    fetch_live_fx_rates as _fetch_live_fx_rates,
    fetch_ticker_meta as _fetch_ticker_meta,
    fetch_ticker_quote_info as _fetch_ticker_quote_info,
    fetch_intraday_sparklines as _fetch_intraday_sparklines,
    fetch_asset_history as _fetch_asset_history,
    fetch_period_reference_price as _fetch_period_reference_price,
)

fetch_price_history          = st.cache_data(ttl=1800, show_spinner=False)(_fetch_price_history)
fetch_price_history_intraday = st.cache_data(ttl=300,  show_spinner=False)(_fetch_price_history_intraday)
fetch_live_quotes            = st.cache_data(ttl=60,   show_spinner=False)(_fetch_live_quotes)
fetch_live_fx_rates          = st.cache_data(ttl=60,   show_spinner=False)(_fetch_live_fx_rates)
fetch_ticker_meta            = st.cache_data(ttl=86400, show_spinner=False)(_fetch_ticker_meta)
fetch_ticker_quote_info      = st.cache_data(ttl=3600, show_spinner=False)(_fetch_ticker_quote_info)
fetch_intraday_sparklines    = st.cache_data(ttl=300,  show_spinner=False)(_fetch_intraday_sparklines)

# Asset detail caching:
# - Historik: kort TTL (intraday skifter, men ikke ved hvert klik)
# - Referencepris: længere TTL (ændrer sig sjældent)
fetch_asset_history = st.cache_data(ttl=300, show_spinner=False)(_fetch_asset_history)
fetch_period_reference_price = st.cache_data(ttl=3600, show_spinner=False)(_fetch_period_reference_price)
                                      
from data.fetch import load_pluto_xlsx_raw as _load_xlsx_raw

@st.cache_data(show_spinner=False)
def load_pluto_xlsx_cached(xlsx_path: str, file_mtime: float):
    return _load_xlsx_raw(xlsx_path, file_mtime)                                                        
                                        

                                                            
                                           
                                                               
                                                            
                                                                         


                                    
                                                                      
                                       
                                                    


                                                   
                
                             
                                                                             
                     
                                              
                                              
                                    
                             
                                                                                   


                                                     
                                       
                                                                  
             

    
                                                                                               

                                                           
                                           
                                                                                         
                                                             

                                            
                                                                                             
                                         
                                                                   

                                     
                                                                      
                                                                                          
                                                                                         
                                                                                                                       
                                                                                                              

                                                    
                                                                                 
                                                                                        
        
                                      
                                          
                                                                                           
         
                       
                                                             
                                                          
         
                                                                 
                                 
                                                                                
                                            
                                                                                                    
                                                                                                    
                                  
                                                  
                                                      
                                   
                                                                   
                                                          
                                 
                                                 
                              
                            
                                                 
                                                                                                   
                                                                       

                                                                                              
                                            
                                            
                                                                   
                                                   
                     
                                                                     
            

               
                                                                                                                            

                                                      
                                                  
                      
                                     
                                         
                                                                                        
         
                                         
                                                                                                  
         
                         
                                                            
                                                                        
                          
         
                                                                    
                                           
                                                                                                  
                                                                                                  
                                                                            
                         
                                              
                                                                   
                                                                   
                                                          
                                                                         
                             
                        
                                                                   
                                                                                    
                                                                     
                    
                                                                                  
                                                                                              
                                                                                              
         
                                                
                            
                                                                     
                                                                     
                                                                     
         
                 

                  
                                                                                                   

                  
                                  
                                                                     
                                                               
                                                        
         

                    
                                                                 
        
                     
                                                                                                
                                                                                                
                                                                                           
                                             
                                                                                                
         
                                                     

                      
                                                 
                   