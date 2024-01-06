import analytics
import pandas as pd 


def load_data(path:str):
    data = pd.read_csv(path)
    return data


def execute(func):

    function = func[0]
    
    if len(func) > 1:
        arg = func[1]
        function(arg)
        return None 

    function()
    return None    
    


if __name__ == "__main__":
    path = input("Path: ")
    if path == "":
        path = "notebook/history.csv"
    dataset = load_data(path)

    port = analytics.Analytics(dataset)
    
    PLOTS = {
        'Equity Curve' : (port.plot_equity_curve,), 
        'Distribution' : (port.plot_returns_distribution,), 
        'P/L Month' : (port.plot_profit_by_time_period, "month"), 
        'P/L Day of Week' : (port.plot_profit_by_time_period, "day_of_week"), 
        'P/L Year' : (port.plot_profit_by_time_period, "year"), 
        'Cumulative Returns by Instrument' : (port.plot_profit_by_instrument,),
        'Cumulative Returns by Position Type' : (port.plot_profit_by_position_type,)
        }

    while True: 
        print()
        print("############ SELECT PLOT ############")
        print()
        print("0. Exit")
        for i, p in enumerate(PLOTS): 
            print(f"{i+1}. {p}")
        
        option = int(input("Select Option: "))

        if option == 0: 
            break
        
        keys = list(PLOTS.keys())

        execute(PLOTS[keys[option - 1]])