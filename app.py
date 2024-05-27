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

st.set_page_config(layout="wide")

# ----------------------------------------------------------------
# functions
# ----------------------------------------------------------------

def backtesting_vix_ls(short_contract, long_contract, multiplier, start_date, end_date):

    res_df = pd.DataFrame(columns=[
        'Date', 'VIX_Price', 'Short_Price', 'Long_Price',
        'Short_Notl', 'Long_Notl', 'Daily_Return'
    ])

    position = False

    def get_next_item(item):
        # Extract the number from the input string
        number = int(item[-1])
        
        # Increment the number by 1
        next_number = number + 1
        
        # Create the next item string
        next_item = f"UX{next_number}"
        
        return next_item

    # Convert start_date and end_date to datetime if they are strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    # Filter the DataFrame based on the date range
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    for _, row in filtered_df.iterrows():
        if row['Date'] in roll_dates:
            # entry condition filter
            if 1 > 0:  # placeholder for entry condition
                short_notl = row[get_next_item(short_contract)] * 100 * -1000
                short_price = row[get_next_item(short_contract)]
                long_notl = row[get_next_item(long_contract)] * 100 * 1000 * multiplier
                long_price = row[get_next_item(long_contract)]
                if not position:  # no prior position to be closed
                    daily_return = 0
                else:
                    # close prior positions
                    this_short_notl = row[short_contract] * 100 * -1000
                    this_long_notl = row[long_contract] * 100 * 1000 * multiplier
                    daily_return = this_short_notl + this_long_notl - (res_df['Short_Notl'].iloc[-1] + res_df['Long_Notl'].iloc[-1])
                position = True
            else:
                short_notl = long_notl = short_price = long_price = daily_return = 0
                position = False
        elif position:  # not roll date but have position
            short_notl = row[short_contract] * 100 * -1000
            short_price = res_df['Short_Price'].iloc[-1]
            long_notl = row[long_contract] * 100 * 1000 * multiplier
            long_price = res_df['Long_Price'].iloc[-1]
            daily_return = short_notl + long_notl - (res_df['Short_Notl'].iloc[-1] + res_df['Long_Notl'].iloc[-1])
        else:  # not roll date and have no position
            short_notl = long_notl = short_price = long_price = daily_return = 0

        res_df.loc[len(res_df)] = [
            row['Date'],
            row['VIX_Spot'],
            short_price,
            long_price,
            short_notl,
            long_notl,
            daily_return
        ]

    
    res_df['Cumulative_PnL'] = res_df['Daily_Return'].cumsum()
    res_df['NAV'] = 1000000 + res_df['Cumulative_PnL']

    # Add a column to track yearly rebalanced NAV
    res_df['Rebalanced_NAV'] = res_df['NAV']

    # Rebalance NAV at the beginning of each year to $1,000,000
    for year in res_df['Date'].dt.year.unique():
        first_date_of_year = res_df[res_df['Date'].dt.year == year].index[0]
        if first_date_of_year != 0:
            initial_nav = res_df.loc[first_date_of_year, 'Rebalanced_NAV']
            res_df.loc[first_date_of_year:, 'Rebalanced_NAV'] -= initial_nav - 1000000

    # Calculate daily percentage returns based on the rebalanced NAV
    res_df['daily_pct_return'] = res_df['Rebalanced_NAV'].pct_change()

    # Set the daily percentage return to 0 at the beginning of each year
    for year in res_df['Date'].dt.year.unique():
        first_date_of_year = res_df[res_df['Date'].dt.year == year].index[0]
        res_df.loc[first_date_of_year, 'daily_pct_return'] = 0

    return res_df

def plot_pnl(df, short_contract, long_contract, multiplier, start_date, end_date):
    fig = go.Figure()
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    start_date = start_date.strftime("%d-%b-%Y")
    end_date = end_date.strftime("%d-%b-%Y")
    # Add a trace for the cumulative PnL data
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_PnL'], mode='lines', 
                             name=f'Cumulative PnL of short 1x {short_contract} and long {multiplier}x {long_contract}, from {start_date} to {end_date}'))

    # Set plot layout
    fig.update_layout(
        xaxis=dict(
            zeroline=True,
            zerolinewidth=10,
            zerolinecolor='black',
            showgrid=True
        ),
        title=f'Cumulative PnL of short 1x {short_contract} and long {multiplier}x {long_contract}, from {start_date} to {end_date}' ,
        xaxis_title='Date',
        yaxis_title='Cumulative PnL in $'
    )

    # Show the figure
    return fig

def plot_pnls(df1, df2=None, df3=None, df4=None, df5=None):
    fig = go.Figure()

    # Add a trace for the cumulative PnL data of the first DataFrame (mandatory)
    fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Cumulative_PnL'], mode='lines', name='Cumulative PnL 1'))

    # Add traces for the optional DataFrames if provided
    if df2 is not None:
        fig.add_trace(go.Scatter(x=df2['Date'], y=df2['Cumulative_PnL'], mode='lines', name='Cumulative PnL 2'))

    if df3 is not None:
        fig.add_trace(go.Scatter(x=df3['Date'], y=df3['Cumulative_PnL'], mode='lines', name='Cumulative PnL 3'))

    if df4 is not None:
        fig.add_trace(go.Scatter(x=df4['Date'], y=df4['Cumulative_PnL'], mode='lines', name='Cumulative PnL 4'))

    if df5 is not None:
        fig.add_trace(go.Scatter(x=df5['Date'], y=df5['Cumulative_PnL'], mode='lines', name='Cumulative PnL 5'))

    # Set plot layout
    fig.update_layout(
        title='Cumulative PnL Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative PnL'    
    )

    # Show the figure
    return fig

def performance_summary_by_year(df):

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Group by year
    df['Year'] = df.index.year

    # Annual return calculation
    annual_return = df.groupby('Year')['daily_pct_return'].apply(lambda x: (1 + x).prod() - 1)

    # Annual volatility calculation
    annual_volatility = df.groupby('Year')['daily_pct_return'].std() * np.sqrt(252)

    # Maximum drawdown calculation
    def max_drawdown(nav):
        roll_max = nav.cummax()
        drawdown = nav / roll_max - 1.0
        return drawdown.min()

    max_drawdown = df.groupby('Year')['Rebalanced_NAV'].apply(max_drawdown)

    # Combine the results into a summary dataframe
    summary = pd.DataFrame({
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Max Drawdown': max_drawdown
    })

    return summary

def performance_summary_full_period(df, risk_free_rate=0.0):
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    dollar_return = df['Cumulative_PnL'][-1]
    max_loss = df['Cumulative_PnL'].min()

    # Calculate the total period return
    total_return = (df['NAV'].iloc[-1] / df['NAV'].iloc[0]) - 1

    # Calculate the annualized return
    n_years = (df.index[-1] - df.index[0]).days / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1

    # Calculate the annualized volatility
    annualized_volatility = df['daily_pct_return'].std() * np.sqrt(252)

    # Calculate the Sharpe ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # Calculate the maximum drawdown
    def max_drawdown(nav):
        roll_max = nav.cummax()
        drawdown = nav / roll_max - 1.0
        return drawdown.min()

    max_drawdown_value = max_drawdown(df['NAV'])

    # Combine the results into a summary dataframe
    summary = pd.DataFrame({
        'Annualized Return': [annualized_return],
        'Annualized Volatility': [annualized_volatility],
        'Sharpe Ratio': [sharpe_ratio],
        'Max Drawdown': [max_drawdown_value],
        'Total $ Return': dollar_return,
        'Max Loss': max_loss
    })

    return summary

# ----------------------------------------------------------------
# data
# ----------------------------------------------------------------

df = pd.read_csv('VIX_Price.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Assume 2 days ahead to roll
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
    "20 Sep 2023", "18 Oct 2023", "15 Nov 2023", "20 Dec 2023",
    "17 Jan 2024", "14 Feb 2024", "20 Mar 2024", "17 Apr 2024"
]
expiration_dates = [datetime.strptime(date, '%d %b %Y') for date in expiration_dates]
#----------------------------------------------------------------
# find roll dates
trade_dates = df['Date'].tolist()
days_ahead = 2
roll_dates = []
for exp_date in expiration_dates:
    roll_date = exp_date - pd.Timedelta(days=days_ahead)
    nearest_date = df.loc[df['Date'] <= roll_date, 'Date'].max()
    if not pd.isnull(nearest_date):
        roll_dates.append(nearest_date)

# Convert roll_dates to datetime.datetime objects
roll_dates = [pd.to_datetime(date) for date in roll_dates]

# ----------------------------------------------------------------
# streamlit UI
# ----------------------------------------------------------------

with st.sidebar:
    st.header('Sell VIX Backtesting')
    start_date = st.date_input('Start Date', date(2017,1,3))
    end_date = st.date_input('End Date', date(2023,12,29))
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    short_contract = st.selectbox('Short Contract',('UX1', 'UX2', 'UX3'), index = 0)
    long_contract = st.selectbox('Long Contract',('UX1', 'UX2', 'UX3'), index = 2)
    multiplier = st.number_input('Multiplier on long contract', value = 1.5, min_value = 0.0, max_value = 10.0)
    run = st.button('Run')

tab1, tab2 = st.tabs(['Single Strategy', 'Strategies Comparison'])
with tab1:
    if run:
        res_df = backtesting_vix_ls(short_contract=short_contract, long_contract=long_contract, multiplier=multiplier, start_date = start_date, end_date = end_date)
        fig1 = plot_pnl(df = res_df, short_contract=short_contract, long_contract=long_contract, multiplier=multiplier, start_date=start_date, end_date=end_date)
        perf_year = performance_summary_by_year(df = res_df)
        perf_all = performance_summary_full_period(df = res_df, risk_free_rate = 0)

        st.plotly_chart(fig1, use_container_width=True)
        # plot vix spot values
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        fig_vix = px.line(filtered_df, x = 'Date', y = 'VIX_Spot', title = 'VIX Spot')
        st.plotly_chart(fig_vix, use_container_width=True)


        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(perf_year)
        with col2:
            st.dataframe(perf_all)

with tab2:
    if run:
        comp_df = pd.DataFrame()

        progress_text = 'Running backtesting.......'
        my_bar = st.progress(0, text=progress_text)
        percent_complete = 0

        res_df_1 = backtesting_vix_ls(short_contract=short_contract, long_contract=long_contract, multiplier=0, start_date = start_date, end_date = end_date)
        comp_df['Pure_short_UX1'] = res_df_1['Cumulative_PnL']
        my_bar.progress(percent_complete + 20, text=progress_text)
        percent_complete += 20

        res_df_2 = backtesting_vix_ls(short_contract=short_contract, long_contract=long_contract, multiplier=0.5, start_date = start_date, end_date = end_date)
        comp_df['Short 1x Long 0.5x'] = res_df_2['Cumulative_PnL']
        my_bar.progress(percent_complete + 20, text=progress_text)
        percent_complete += 20
        
        res_df_3 = backtesting_vix_ls(short_contract=short_contract, long_contract=long_contract, multiplier=1, start_date = start_date, end_date = end_date)
        comp_df['Short 1x Long 1x'] = res_df_3['Cumulative_PnL']
        my_bar.progress(percent_complete + 20, text=progress_text)
        percent_complete += 20

        res_df_4 = backtesting_vix_ls(short_contract=short_contract, long_contract=long_contract, multiplier=1.5, start_date = start_date, end_date = end_date)
        comp_df['Short 1x Long 1.5x'] = res_df_4['Cumulative_PnL']
        my_bar.progress(percent_complete + 20, text=progress_text)
        percent_complete += 20

        res_df_5 = backtesting_vix_ls(short_contract=short_contract, long_contract=long_contract, multiplier=2, start_date = start_date, end_date = end_date)
        comp_df['Short 1x Long 2x'] = res_df_5['Cumulative_PnL']
        my_bar.progress(percent_complete + 20, text=progress_text)
        percent_complete += 20
        my_bar.empty()

        comp_df['Date'] = res_df_5['Date']


        fig = px.line(comp_df, x = 'Date', y = comp_df.columns,
                hover_data = {'Date': '|%B %d, %Y'},
                title = 'Compare cumulative return across different spread ratios')
        fig.update_xaxes(
            ticklabelmode = 'period'
        )

        st.plotly_chart(fig, theme = 'streamlit', use_container_width = True)
        
        comp_df.columns.names = ['Spread_Ratio']
        comp_df.set_index('Date', inplace = True)
        fig2 = px.area(comp_df, facet_col = 'Spread_Ratio', facet_col_wrap = 2)
        st.plotly_chart(fig2, theme = 'streamlit', use_container_width = True)
