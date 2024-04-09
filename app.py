import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import norm
from datetime import datetime, timedelta
from datetime import date
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.optimize import fsolve

# Functions
def black_scholes(S, K, T, r, sigma, option_type="call"):

    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":   
        option_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "put":
        option_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return option_price

def black_scholes_delta(S, K, T, r, sigma, option_type="call"):

    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    
    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1  # or equivalently -norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")
    
    return delta

# Function to find the strike price for a given delta
def find_strike_for_delta(S, T, r, sigma, delta, option_type='call'):
    
    # Define the inverse of the Black-Scholes Delta function for call options to solve for K
    def bs_inverse_call_delta(K, S, T, r, sigma, delta):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) - delta

    # Define the inverse of the Black-Scholes Delta function for put options to solve for K
    def bs_inverse_put_delta(K, S, T, r, sigma, delta):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(-d1) - delta
    
    # Initial guess for K
    K_guess = S
    if option_type == 'call':
        K = fsolve(bs_inverse_call_delta, K_guess, args=(S, T, r, sigma, delta))
    elif option_type == 'put':
        K = fsolve(bs_inverse_put_delta, K_guess, args=(S, T, r, sigma, delta))
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    return K[0]

def format_number(number):
    abs_number = abs(number)
    if abs_number >= 1000000:
        formatted_number = f"{number / 1000000:.2f}m"
    elif abs_number >= 1000:
        formatted_number = f"{number / 1000:.2f}k"
    else:
        formatted_number = f"{number:.2f}"
    
    if number < 0:
        return f"$ {formatted_number}"
    else:
        return f"$ {formatted_number}"

#----------------------------------------------------------------
# Functions
def get_front_month_column(trade_date, roll_day):
    adjusted_trade_date = trade_date + timedelta(days=roll_day)
    for i in range(len(expiration_dates)):
        if adjusted_trade_date >=  expiration_dates[i - 1] and adjusted_trade_date < expiration_dates[i]:
            return(expiration_mapping[expiration_dates[i]])

def get_next_month_abbr(date_obj):
    # Add one month to the date
    next_month_obj = date_obj + timedelta(days=30)
    
    # Return the abbreviated month name
    return next_month_obj.strftime("%b")

def get_this_month_abbr(date_obj):
    this_month_abbr = date_obj

    return this_month_abbr.strftime("%b")

def find_interval(lst, x):
    for i in range(1, len(lst)):
        if lst[i-1] <= x < lst[i]:
            return lst[i]
    return None

    
    return delta

def skewness(VIX_spot):
    SPX_delta_put_key = list(range(1,51,1))
    SPX_skewness_put_value = []
    for i in SPX_delta_put_key:
        if i <= 25:
            SPX_skewness_put_value.append((25 - i) * 0.2 + VIX_spot)
        elif i > 25:
            SPX_skewness_put_value.append(VIX_spot - (i-25) * 0.2)

    SPX_delta_call_key = list(range(1,51,1))
    SPX_skewness_call_value = [ SPX_skewness_put_value[-1] for _ in range(50)]

    SPX_put_skewness = dict(zip(SPX_delta_put_key, SPX_skewness_put_value))
    SPX_call_skewness = dict(zip(SPX_delta_call_key, SPX_skewness_call_value))

    return(SPX_put_skewness,SPX_call_skewness)

def find_strike_for_delta(S, T, r, sigma, delta, option_type='call'):
    
    # Define the inverse of the Black-Scholes Delta function for call options to solve for K
    def bs_inverse_call_delta(K, S, T, r, sigma, delta):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) - delta

    # Define the inverse of the Black-Scholes Delta function for put options to solve for K
    def bs_inverse_put_delta(K, S, T, r, sigma, delta):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(-d1) - delta
    
    # Initial guess for K
    K_guess = S
    if option_type == 'call':
        K = fsolve(bs_inverse_call_delta, K_guess, args=(S, T, r, sigma, delta))
    elif option_type == 'put':
        K = fsolve(bs_inverse_put_delta, K_guess, args=(S, T, r, sigma, delta))
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    return K[0]

def backtest(day_to_roll = 2, buy_SPX = True, SPX_delta = 25, investment_percentage = 0.5, carry_trigger = 0.05, start_date = None, end_date = None):

    #----------------------------------------------------------------
    # define day to roll
    dtr = day_to_roll

    vix_df = pd.read_csv('VIX.csv')
    vix_df['Date'] = pd.to_datetime(vix_df['Date'], format='%d/%m/%Y')

    for column in vix_df.columns:
        if column == 'Date':
            continue
        else:
            vix_df[column] = pd.to_numeric(vix_df[column], errors='coerce')

    #----------------------------------------------------------------
    # Expiration dates for the VIX futures
    expiration_dates = [
        "18 Jan 2017", "15 Feb 2017", "22 Mar 2017", "19 Apr 2017",
        "17 May 2017", "21 Jun 2017", "19 Jul 2017", "16 Aug 2017",
        "20 Sep 2017", "18 Oct 2017", "15 Nov 2017", "20 Dec 2017",
        "17 Jan 2018", "14 Feb 2018", "21 Mar 2018", "18 Apr 2018",
        "16 May 2018", "20 Jun 2018", "18 Jul 2018", "22 Aug 2018",
        "19 Sep 2018", "17 Oct 2018", "21 Nov 2018", "19 Dec 2018",
        "16 Jan 2019", "13 Feb 2019", "19 Mar 2019", "17 Apr 2019",
        "22 May 2019", "19 Jun 2019", "17 Jul 2019", "21 Aug 2019",
        "18 Sep 2019", "16 Oct 2019", "20 Nov 2019", "18 Dec 2019",
        "22 Jan 2020", "19 Feb 2020", "18 Mar 2020", "15 Apr 2020",
        "20 May 2020", "17 Jun 2020", "22 Jul 2020", "19 Aug 2020",
        "16 Sep 2020", "21 Oct 2020", "18 Nov 2020", "16 Dec 2020",
        "20 Jan 2021", "17 Feb 2021", "17 Mar 2021", "21 Apr 2021",
        "19 May 2021", "16 Jun 2021", "21 Jul 2021", "18 Aug 2021",
        "15 Sep 2021", "20 Oct 2021", "17 Nov 2021", "22 Dec 2021",
        "19 Jan 2022", "16 Feb 2022", "15 Mar 2022", "20 Apr 2022",
        "18 May 2022", "15 Jun 2022", "20 Jul 2022", "17 Aug 2022",
        "21 Sep 2022", "19 Oct 2022", "16 Nov 2022", "21 Dec 2022",
        "18 Jan 2023", "15 Feb 2023", "22 Mar 2023", "19 Apr 2023",
        "17 May 2023", "21 Jun 2023", "19 Jul 2023", "16 Aug 2023",
        "20 Sep 2023", "18 Oct 2023", "15 Nov 2023", "20 Dec 2023"
    ]
    #----------------------------------------------------------------
    # Convert expiration dates to datetime objects
    expiration_dates = [datetime.strptime(date, '%d %b %Y') for date in expiration_dates]

    # Map expiration dates to their corresponding month column
    expiration_mapping = {date: date.strftime('%b') for date in expiration_dates}

    #----------------------------------------------------------------
    # find roll date
    roll_date = [x - timedelta(dtr) for x in expiration_dates]
    all_date = [x.to_pydatetime() for x in vix_df['Date']]
    new_roll_date = []
    for r in roll_date:
        while r not in all_date:
            r = r - timedelta(1)
        new_roll_date.append(r)

    #----------------------------------------------------------------
    # backtest
    # assumptions
    contract_size = 100
    carry_trigger = carry_trigger
    spx_put_delta = SPX_delta
    r = 0.05
    investment_percentage = investment_percentage
    result = []    
    open = False
    buy_SPX_put = buy_SPX

    put_pnl = {}
    vix_pnl = {}
    carry_pnl = {}

    start_date_index = vix_df.index[vix_df['Date'] == start_date].tolist()
    end_date_index = vix_df.index[vix_df['Date'] == end_date].tolist()

    # Check if start_date or end_date were not found and handle accordingly
    start_date_index = start_date_index[0] if start_date_index else 0  # Set to 0 if not found
    end_date_index = end_date_index[0] if end_date_index else len(vix_df) - 1  # Set to last index if not found


    for i in range(len(vix_df)):
        if i >= start_date_index and i <= end_date_index:
            this_date = vix_df.iloc[i]['Date']
            ###
            ### if short VIX now, find the expected carry which is today's future price - VIX spot price
            contract = get_next_month_abbr(this_date) # find which month contract it is
            expected_carry = vix_df.iloc[i][contract] - vix_df.iloc[i]['VIX_Spot']
            expected_carry_percentage = vix_df.iloc[i][contract] / vix_df.iloc[i]['VIX_Spot'] - 1
            expected_carry_notional = expected_carry * contract_size * 1000
            
            ###
            if this_date in new_roll_date and expected_carry_percentage >= carry_trigger: # it is time to roll
                open = True
                this_carry = expected_carry_notional
                carry_pnl[this_date] = this_carry
                ### VIX futures position
                this_price = vix_df.iloc[i][contract]
                entry_date = this_date
                entry_price = vix_df.iloc[i][contract]

                short_position_notional = this_price * contract_size * 1000 * -1

                # this VIX futures expiration date
                exp_date = find_interval(lst = new_roll_date, x = this_date)
                dte = (exp_date - this_date).days / 360

                ### Buy SPX put while we roll VIX futures
                if buy_SPX_put:
                    current_spx_spot = vix_df.iloc[i]['SPX_Spot']
                    SPX_put_skewness = skewness(VIX_spot=vix_df.iloc[i]['VIX_Spot'])[0]
                    SPX_vol = SPX_put_skewness[spx_put_delta] / 100
                    SPX_strike = find_strike_for_delta(S = current_spx_spot, T = dte, r = r, sigma = SPX_vol, delta = spx_put_delta / 100, option_type='put')
                    SPX_put_price = black_scholes(S = current_spx_spot, K = SPX_strike, T = dte, r = r, sigma=SPX_vol, option_type='put')
                    ### SPX put size depends on the expected carry of VIX futures
                    SPX_size = expected_carry_notional * investment_percentage / SPX_put_price / 100
                    SPX_notional = SPX_put_price * SPX_size * 100
                else:
                    SPX_put_price = 0
                    SPX_size = 0
                    SPX_notional = 0

                daily_return = 0

                put_daily_pnl = 0
                short_daily_pnl = 0
                this_carry = this_carry

                result.append([this_date, entry_price, this_price, put_daily_pnl, short_daily_pnl, this_carry,
                                contract_size, short_position_notional,SPX_put_price, SPX_size, SPX_notional, daily_return])
            
            elif this_date in new_roll_date and expected_carry_percentage < carry_trigger: # skip this roll
                open = False
                

            else: # it is not time to roll
                if open: # we have a position open
                    ### VIX futures position
                    this_date = vix_df.iloc[i]['Date']
                    which_expiration_date = find_interval(lst = new_roll_date, x = this_date)
                    if which_expiration_date is None:
                        break
                    contract = get_this_month_abbr(which_expiration_date)
                    this_price = vix_df.iloc[i][contract]
                    last_price = vix_df.iloc[i-1][contract]
                    short_position_notional = contract_size * this_price * -1000
                    short_daily_pnl = (last_price - this_price) * contract_size * 1000

                    ### put position
                    if buy_SPX_put:
                        current_spx_spot = vix_df.iloc[i]['SPX_Spot']
                        SPX_put_skewness = skewness(VIX_spot=vix_df.iloc[i]['VIX_Spot'])[0]
                        # SPX_vol = SPX_put_skewness[spx_put_delta] / 100
                        SPX_vol = vix_df.iloc[i]['VIX_Spot'] / 100
                        dte = (exp_date - this_date).days / 360
                        if dte == 0:
                            dte = 1 / 360
                        SPX_put_price = black_scholes(S = current_spx_spot, K = SPX_strike, T = dte, r = r, sigma=SPX_vol, option_type='put')
                        SPX_notional = SPX_put_price * 100 * result[-1][-3]
                        put_daily_pnl = SPX_notional - result[-1][-2]

                    else:
                        SPX_put_price = 0
                        SPX_size = 0
                        SPX_notional = 0
                        put_daily_pnl = 0
                    ### consolidate
                    daily_return = short_daily_pnl + put_daily_pnl

                    put_pnl[this_date] = put_daily_pnl
                    vix_pnl[this_date] = short_daily_pnl
                    carry_pnl[this_date] = this_carry

                    result.append([this_date, entry_price, this_price, put_daily_pnl, short_daily_pnl, this_carry,
                                     contract_size, short_position_notional,SPX_put_price, SPX_size, SPX_notional, daily_return])


    #----------------------------------------------------------------
    # handle the result dataframe
    res = pd.DataFrame(data=result, columns=['Trade_Date','Entry_Price', 'Today_Price', 'Put_Daily_PnL', 'VIX_Short_Daily_PnL'
                                             , 'Carry_Daily_PnL','Contract_Size', 'Short_Position_Notional','SPX_Put_Price','SPX_Size', 'SPX_Notional', 'Daily_PnL'])               

    # Calculate the 'Cum_Return' column
    initial_investment = 1400000
    res['Cum_Dollar_PnL'] = res['Daily_PnL'].expanding(min_periods=1).sum()
    res['NAV'] = res['Cum_Dollar_PnL']  + initial_investment
    res['Daily_Return'] = res['NAV'].pct_change()
    res['Cum_Return'] = res['NAV'] / initial_investment - 1

    return [res,[put_pnl, vix_pnl, carry_pnl]]

def metrics(res):

    res['running_max'] = res['Cum_Return'].cummax()
    res['Drawdown'] = res['running_max'] - res['Cum_Return']
    res['max_drawdown'] = res['Drawdown'].cummax()
    res.drop(columns=['running_max'], inplace=True)

################################################################################################
    mean_daily_return = res['Daily_Return'].mean()
    max_daily_return = res['Daily_Return'].max()
    min_daily_return = res['Daily_Return'].min()
    std_daily_return = res['Daily_Return'].std()
    max_drawdown = res['max_drawdown'].max()
    min_pnl = res['Cum_Dollar_PnL'].min()
    max_pnl = res['Cum_Dollar_PnL'].max()
    pnl = res['Cum_Dollar_PnL'].iloc[-1]

    metrics_list = [
        mean_daily_return,
        max_daily_return,
        min_daily_return,
        std_daily_return,
        max_drawdown,
        min_pnl,
        max_pnl,
        pnl
    ]
    
    return metrics_list

def plot_pnl(res):
    fig = px.line(res, x='Trade_Date', y='Cum_Dollar_PnL', title='Cum_Dollar_PnL ')

    # Customize the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Cum_Dollar_PnL',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    # Show the plot
    st.plotly_chart(fig)

def plot_time_series(df, title = ' Cumulative PnL'):
    date_column='Date'
    value_column='Cum_PnL'

    fig = px.line(df, x=date_column, y=value_column, title=title)
    fig.update_xaxes(title=date_column)
    fig.update_yaxes(title=value_column)
    fig.update_layout(showlegend=False)
    return fig

def consolidate_df(d):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(d, orient='index', columns=['PnL'])

    # If you want to reset the index to make the Timestamps a column, you can do the following:
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    df['Cum_PnL'] = df['PnL'].cumsum()
    return df

#----------------------------------------------------------------
# tabs
tab1, tab2,tab3 = st.tabs(['Calculator', 'Optimization','Backtesting'])

with tab1:
    #----------------------------------------------------------------
    # VIX Assumptions
    with st.container(border=True):
        st.write('VIX Parameters')
        col1, col2, col3 = st.columns(3)
        with col1:
            VIX_spot = st.number_input('VIX Spot Price', value = 14.00)
        with col2:
            VIX_carry = st.number_input('VIX Carry', 1.2) # points per month
        with col3:
            VIX_size = st.number_input('VIX Contract Size', 100) # number of contracts we sell
        col1, col2 = st.columns(2)
        with col1:
            investment_percentage = st.slider('Investment Percentage (%)', 1,100,50) / 100

        with col2:
            VIX_beta = st.slider('VIX Beta', 1,40,3)


    #----------------------------------------------------------------
    # Set up SPX vol curve
    # linear model
    # assume VIX current spot = SPX 25 delta put
    # assume with 1 change in delta IV changes 0.2v
    SPX_delta_put_key = list(range(1,51,1))
    SPX_skewness_put_value = []
    for i in SPX_delta_put_key:
        if i <= 25:
            SPX_skewness_put_value.append((25 - i) * 0.2 + VIX_spot)
        elif i > 25:
            SPX_skewness_put_value.append(VIX_spot - (i-25) * 0.2)

    SPX_delta_call_key = list(range(1,51,1))
    SPX_skewness_call_value = [ SPX_skewness_put_value[-1] for _ in range(50)]

    SPX_put_skewness = dict(zip(SPX_delta_put_key, SPX_skewness_put_value))
    SPX_call_skewness = dict(zip(SPX_delta_call_key, SPX_skewness_call_value))

    #----------------------------------------------------------------
    # SPX Assumptions
    with st.container(border=True):
        st.write('SPX Parameters')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            SPX_spot = st.number_input('SPX Spot Price', 5200)
        with col2:
            SPX_delta = st.number_input('SPX Put Delta', 1,50,25)
        with col3:
            days = st.number_input('Days to Expire', 30)
            dte = days / 360
        with col4:
            r = st.number_input('Interest Rate', 0.05)
        
        SPX_vol = SPX_put_skewness[SPX_delta] / 100
        SPX_Strike = find_strike_for_delta(S = SPX_spot, T = dte, r = r, sigma = SPX_vol, delta = SPX_delta / 100, option_type='put')
        SPX_put_price = black_scholes(S = SPX_spot, K = SPX_Strike, T = dte, r = r, sigma=SPX_vol, option_type='put')

    #----------------------------------------------------------------
    # generate result with given SPX delta
    monthly_revenue = VIX_carry * VIX_size * 1000
    SPX_notl_to_buy = monthly_revenue * investment_percentage / SPX_put_price * SPX_spot
    SPX_delta_notl = SPX_notl_to_buy * -1 * SPX_delta / 100
    VIX_SPX_notl = VIX_spot * VIX_size * VIX_beta * 1000
    remaining_delta = SPX_delta_notl + VIX_SPX_notl

    if st.button('Calculate'):
        st.divider()

        col1, col2, col3= st.columns(3)
        col1.metric("SPX Put Strike", round(SPX_Strike,0))
        col2.metric('SPX Put Price',round(SPX_put_price,2))
        col3.metric('SPX Put IV', format_number(SPX_vol))


        col1, col2, col3= st.columns(3)
        col1.metric("Carry Left", format_number(monthly_revenue * (1-investment_percentage))) 
        col2.metric('SPX Delta Notl',format_number(SPX_delta_notl))
        col3.metric('Remaining Delta', format_number(remaining_delta))

with tab2:
    remaining_delta_list = []
    delta = list(range(1,51,1))

    for i in range(1,51,1):
        SPX_delta = i
        SPX_vol = SPX_put_skewness[SPX_delta] / 100
        SPX_delta = i / 100
        SPX_Strike = find_strike_for_delta(S = SPX_spot, T = dte, r = r, sigma = SPX_vol, delta = SPX_delta, option_type='put')
        SPX_notl_to_buy = monthly_revenue * investment_percentage / black_scholes(S = SPX_spot, K = SPX_Strike, T = dte, r = r, sigma=SPX_vol, option_type='put') * SPX_spot
        SPX_delta_notl = SPX_notl_to_buy * black_scholes_delta(S = SPX_spot, K = SPX_Strike, T = dte, r = r, sigma=SPX_vol, option_type='put')
        remaining_delta = SPX_delta_notl + VIX_SPX_notl
        remaining_delta_list.append(round(remaining_delta,0))

    df = pd.DataFrame(data=[remaining_delta_list], columns = delta)

    # Find the first column with a value less than 0
    mask = df.lt(0)  # Create a boolean mask where each cell is True if the value is less than 0
    first_col_idx = mask.idxmax(axis=1)  # Get the index of the first occurrence of True for each row
    first_column = first_col_idx.iloc[0] if mask.any(axis=1).iloc[0] else None
    
    st.write(first_column)
    st.write(SPX_spot)

    y_values = df.iloc[0].values
    x_values = df.columns.astype(str)

    # Create the figure
    fig = go.Figure(data=[
        go.Line(x=x_values, y=y_values)
    ])

    # Update the layout if needed
    fig.update_layout(
        title="Remaining Delta by Strike Delta",
        xaxis_title="Strike Delta",
        yaxis_title="Remaining Delta",
        # You can add more customization here if needed
    )

    st.plotly_chart(fig)

with tab3:
    
    buy_SPX = st.checkbox('Buy SPX Put', True)
    day_to_roll = st.number_input('Days Ahead to Roll', min_value=1, max_value=7, value=2)
        
    col1, col2, col3 = st.columns(3)
    with col1:
        SPX_delta = st.number_input('SPX put delta', min_value=1, max_value=50, value=25)
    with col2:
        investment_percentage = st.number_input('Investment Percentage', min_value=0.01, max_value=1.00, value=0.5)
    with col3:
        carry_trigger = st.number_input('Carry Threshold', min_value=-1.00, max_value=1.00, value=0.05)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Backtest Start Date",date(2017, 1, 3))
        start_date = pd.to_datetime(start_date)
    with col2:
        end_date = st.date_input("Backtest End Date",date(2023, 12, 29))
        end_date = pd.to_datetime(end_date)

    if st.button('Run'):
        b = backtest(day_to_roll=day_to_roll, buy_SPX=buy_SPX, SPX_delta = SPX_delta, investment_percentage = investment_percentage, 
                     carry_trigger = carry_trigger, start_date=start_date, end_date = end_date)
        
        x = b[0]
        put_pnl = b[1][0]
        vix_pnl = b[1][1]
        carry_pnl = b[1][2]
        put_pnl_df = consolidate_df(d = put_pnl)
        vix_pnl_df = consolidate_df(d = vix_pnl)        
        carry_pnl_df = consolidate_df(d = carry_pnl)
        put_vix_pnl_df = pd.DataFrame({
            'Date': x['Trade_Date'],
            'Cum_PnL': x['Cum_Dollar_PnL'] - carry_pnl_df['Cum_PnL']
        })
        # Initialize the 'Cum_PnL' column with zeros
        carry_pnl_df['Cum_PnL'] = 0.0

        # Iterate through the DataFrame rows
        for i in range(len(carry_pnl_df)):
            if i == 0:
                # For the first row, 'Cum_PnL' is just 'PnL'
                carry_pnl_df.iloc[i, carry_pnl_df.columns.get_loc('Cum_PnL')] = carry_pnl_df.iloc[i]['PnL']
            else:
                # For subsequent rows, add 'PnL' to 'Cum_PnL' only if 'PnL' has changed from the previous day
                if carry_pnl_df.iloc[i]['PnL'] != carry_pnl_df.iloc[i - 1]['PnL']:
                    carry_pnl_df.iloc[i, carry_pnl_df.columns.get_loc('Cum_PnL')] = carry_pnl_df.iloc[i - 1]['Cum_PnL'] + carry_pnl_df.iloc[i]['PnL']
                else:
                    # If 'PnL' hasn't changed, 'Cum_PnL' remains the same as the previous day
                    carry_pnl_df.iloc[i, carry_pnl_df.columns.get_loc('Cum_PnL')] = carry_pnl_df.iloc[i - 1]['Cum_PnL']

        vix_pnl_df['Cum_PnL'] = vix_pnl_df['Cum_PnL'] - carry_pnl_df['Cum_PnL']
        
        put_vix_pnl_df = pd.DataFrame({
        'Date': x['Trade_Date'],
        'Cum_PnL': x['Cum_Dollar_PnL'] - carry_pnl_df['Cum_PnL']
        })

        plot_pnl(res = x)

        put = plot_time_series(df = put_pnl_df, title='Put PnL')
        st.plotly_chart(put)
        vix = plot_time_series(df = vix_pnl_df, title='VIX Short Component PnL')
        st.plotly_chart(vix)
        carry = plot_time_series(df = carry_pnl_df, title='Carry PnL')
        st.plotly_chart(carry)
        put_vix = plot_time_series(df = put_vix_pnl_df, title='VIX Short Component + Put PnL')
        st.plotly_chart(put_vix)

        metrics_values = metrics(res = x)
        metrics_df = pd.DataFrame(metrics_values, index=[
            'Mean Daily Return (%)',
            'Max Daily Return (%)',
            'Min Daily Return (%)',
            'Std Daily Return (%)',
            'Max Drawdown',
            'Min PnL',
            'Max PnL',
            'PnL'
        ], columns=['Value'])
        st.dataframe(metrics_df)

        # x.to_csv('full.csv')
        
