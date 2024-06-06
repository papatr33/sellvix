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

def backtesting_vix_ls(short_contract, long_contract, multiplier, start_date, end_date, carry_threshold, spread_threshold, stop_loss):

    res_df = pd.DataFrame(columns=[
        'Date', 'VIX_Price', 'Short_Price', 'Long_Price',
        'Short_Notl', 'Long_Notl', 'Daily_Return','Daily_Pct_Return','Short_Carry','Long_Carry'
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

        # check if vol has reach a stop loss level
        if stop_loss and position and row['VIX_Spot'] > stop_loss:
            # unwind current positions
            this_short_notl = row[short_contract] * 100 * -1000
            this_long_notl = row[long_contract] * 100 * 1000 * multiplier
            daily_return = this_short_notl + this_long_notl - (res_df['Short_Notl'].iloc[-1] + res_df['Long_Notl'].iloc[-1])
            daily_pct_return = daily_return /  (abs(res_df['Short_Notl'].iloc[-1]) + abs(res_df['Long_Notl'].iloc[-1]))

            position = False
        
            continue

        if row['Date'] in roll_dates:
            current_roll_date = row['Date']
            date_index = roll_dates.index(current_roll_date)
            next_date = expiration_dates[date_index]

            expected_short_carry = row[get_next_item(short_contract)] / row['VIX_Spot'] - 1
            expected_long_carry = row[get_next_item(long_contract)] / row[get_next_item(short_contract)] - 1
            net_carry = expected_short_carry - expected_long_carry
            expected_spread = row[get_next_item(long_contract)] - row[get_next_item(short_contract)]

            if carry_threshold:
                entry_condition = net_carry > carry_threshold 
            elif spread_threshold:
                entry_condition = expected_spread < spread_threshold
            else:
                entry_condition = True
             
            # entry condition filter
            if entry_condition:  
                short_notl = row[get_next_item(short_contract)] * 100 * -1000
                short_price = row[get_next_item(short_contract)]
                long_notl = row[get_next_item(long_contract)] * 100 * 1000 * multiplier
                long_price = row[get_next_item(long_contract)]
                short_carry = short_price / row['VIX_Spot'] - 1
                long_carry = long_price / short_price - 1
                if not position:  # no prior position to be closed
                    daily_return = 0
                    daily_pct_return = 0
                else:
                    # close prior positions
                    this_short_notl = row[short_contract] * 100 * -1000
                    this_long_notl = row[long_contract] * 100 * 1000 * multiplier
                    daily_return = this_short_notl + this_long_notl - (res_df['Short_Notl'].iloc[-1] + res_df['Long_Notl'].iloc[-1])
                    daily_pct_return = daily_return /  (abs(res_df['Short_Notl'].iloc[-1]) + abs(res_df['Long_Notl'].iloc[-1]))

                position = True
                
            else:
                short_notl = long_notl = short_price = long_price = daily_return = daily_pct_return = short_carry = long_carry = 0
                position = False
        elif position:  # not roll date but have position
            # check if today is between roll date and price switch date
            if current_roll_date < row['Date'] <= next_date:
                short_notl = row[get_next_item(short_contract)] * 100 * -1000
                short_price = res_df['Short_Price'].iloc[-1]
                long_notl = row[get_next_item(long_contract)] * 100 * 1000 * multiplier
                long_price = res_df['Long_Price'].iloc[-1]
            else:
                short_notl = row[short_contract] * 100 * -1000
                short_price = res_df['Short_Price'].iloc[-1]
                long_notl = row[long_contract] * 100 * 1000 * multiplier
                long_price = res_df['Long_Price'].iloc[-1]
            short_end_carry = short_price / row['VIX_Spot'] - 1
            long_end_carry = long_price / short_price - 1
            daily_return = short_notl + long_notl - (res_df['Short_Notl'].iloc[-1] + res_df['Long_Notl'].iloc[-1])
            daily_pct_return = daily_pct_return = daily_return /  (abs(res_df['Short_Notl'].iloc[-1]) + abs(res_df['Long_Notl'].iloc[-1]))

        else:  # not roll date and have no position
            short_notl = long_notl = short_price = long_price = daily_return = daily_pct_return = short_carry = long_carry = 0

        res_df.loc[len(res_df)] = [
            row['Date'],
            row['VIX_Spot'],
            short_price,
            long_price,
            short_notl,
            long_notl,
            daily_return,
            daily_pct_return,
            short_carry,
            long_carry
        ]
    
    res_df['Cumulative_PnL'] = res_df['Daily_Return'].cumsum()
    res_df['NAV'] = 1000000 + res_df['Cumulative_PnL']
    res_df['Cumulative_Pct_PnL'] = (1 + res_df['Daily_Pct_Return']).cumprod() - 1

    res_df['Net_Carry'] = res_df['Short_Carry'] - res_df['Long_Carry']

    return res_df

def plot_pnl(dfs, short_contracts, long_contracts, multipliers, carry_thresholds, spread_thresholds, stop_losses, start_date, end_date):
    fig = go.Figure()

    for i in range(len(dfs)):
        df = dfs[i]
        short_contract = short_contracts[i]
        long_contract = long_contracts[i]
        multiplier = multipliers[i]
        carry_threshold = carry_thresholds[i]
        spread_threshold = spread_thresholds[i]
        stop_loss = stop_losses[i]
        # vix_threshold = vix_thresholds[i]

        trace_name = f'Short {short_contract} Long {multiplier}x {long_contract}, Net Carry > {carry_threshold}, Spread < {spread_threshold}, SL trigger = {stop_loss}'

        trace = go.Scatter(
            x=df['Date'],
            y=df['Cumulative_PnL'],
            mode='lines',
            name=trace_name,
            showlegend=True  # Set showlegend to True for each trace
        )
        fig.add_trace(trace)

    start_date_str = start_date.strftime("%d-%b-%Y")
    end_date_str = end_date.strftime("%d-%b-%Y")

    # Set plot layout
    fig.update_layout(
        xaxis=dict(
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='black',
            showgrid=True
        ),
        title=f'Cumulative PnL from {start_date_str} to {end_date_str}',
        xaxis_title='Date',
        yaxis_title='Cumulative PnL in $',
        legend_title='Contracts',
        legend=dict(
            x=1.02,  # Adjust the x-position of the legend
            y=1,  # Adjust the y-position of the legend
            xanchor='left',  # Anchor the legend to the left side
            yanchor='top',  # Anchor the legend to the top
            bordercolor='black',
            borderwidth=0
        ),
        margin=dict(r=400)  # Increase the right margin to make space for the legend
    )

    # Show the figure
    return fig

def performance_summary_by_year(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Group by year
    df['Year'] = df.index.year

    # Calculate the number of days with positions for each year
    days_with_positions = df.groupby('Year')['Short_Notl'].apply(lambda x: (x != 0).sum())

    # Annual return calculation
    annual_return = df.groupby('Year')['Daily_Pct_Return'].apply(lambda x: (1 + x).prod() - 1)

    # Expected annual return = mean of daily % return * days_with_positions
    # mean_return = df.groupby('Year')['Daily_Pct_Return'].mean() * days_with_positions

    # Annual volatility calculation
    annual_volatility = df.groupby('Year')['Daily_Pct_Return'].std() * np.sqrt(days_with_positions)

    # Sharpe Ratio
    sharpe_ratio = annual_return / annual_volatility

    # Maximum drawdown calculation
    def calculate_max_drawdown(cumulative_pct_return):
        roll_max = cumulative_pct_return.cummax()
        drawdown = cumulative_pct_return - roll_max
        return drawdown.min()

    max_drawdown = df.groupby('Year')['Cumulative_Pct_PnL'].apply(calculate_max_drawdown)

    # Combine the results into a summary dataframe
    summary = pd.DataFrame({
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    })

    # Calculate overall performance metrics
    total_days_with_positions = df['Short_Notl'].count()
    overall_return = (1 + df['Daily_Pct_Return']).prod() - 1
    daily_return = overall_return / total_days_with_positions
    daily_volatility = df['Daily_Pct_Return'].std()
    overall_sharpe_ratio = daily_return / daily_volatility * np.sqrt(252)
    overall_max_drawdown = calculate_max_drawdown(df['Cumulative_Pct_PnL'])

    # Create a new row with overall performance metrics
    overall_performance = pd.DataFrame({
        'Annual Return': [overall_return / df['Year'].nunique()],
        'Annual Volatility': [daily_volatility * np.sqrt(252)],
        'Sharpe Ratio': [overall_sharpe_ratio],
        'Max Drawdown': [overall_max_drawdown]
    }, index=['Overall'])

    # Concatenate the overall performance row to the summary dataframe
    summary = pd.concat([summary, overall_performance])

    return summary

def plot_curves(date1, date2, df):
    # Extract the values from the DataFrame based on the first date
    row1 = df.loc[df['Date'] == date1]
    vix_spot1 = row1['VIX_Spot'].iloc[0]
    vix_1_1 = row1['UX1'].iloc[0]
    vix_2_1 = row1['UX2'].iloc[0]
    vix_3_1 = row1['UX3'].iloc[0]
    vix_4_1 = row1['UX4'].iloc[0]

    # Extract the values from the DataFrame based on the second date
    row2 = df.loc[df['Date'] == date2]
    vix_spot2 = row2['VIX_Spot'].iloc[0]
    vix_1_2 = row2['UX1'].iloc[0]
    vix_2_2 = row2['UX2'].iloc[0]
    vix_3_2 = row2['UX3'].iloc[0]
    vix_4_2 = row2['UX4'].iloc[0]

    # Create the trace for the first curve
    trace1 = go.Scatter(
        x=['Spot', '1M', '2M', '3M', '4M'],
        y=[vix_spot1, vix_1_1, vix_2_1, vix_3_1, vix_4_1],
        mode='lines+markers',
        name=f'VIX Curve - {date1}',
        marker=dict(size=8)
    )

    # Create the trace for the second curve
    trace2 = go.Scatter(
        x=['Spot', '1M', '2M', '3M', '4M'],
        y=[vix_spot2, vix_1_2, vix_2_2, vix_3_2, vix_4_2],
        mode='lines+markers',
        name=f'VIX Curve - {date2}',
        marker=dict(size=8)
    )

    # Calculate the differences between the two curves
    diff_spot = vix_spot2 - vix_spot1
    diff_1m = vix_1_2 - vix_1_1
    diff_2m = vix_2_2 - vix_2_1
    diff_3m = vix_3_2 - vix_3_1
    diff_4m = vix_4_2 - vix_4_1

    # Create annotations for the differences
    annotations = [
        dict(x='Spot', y=vix_spot2, xref='x', yref='y', text=f'Diff: {diff_spot:.2f}', showarrow=True, arrowhead=1),
        dict(x='1M', y=vix_1_2, xref='x', yref='y', text=f'Diff: {diff_1m:.2f}', showarrow=True, arrowhead=1),
        dict(x='2M', y=vix_2_2, xref='x', yref='y', text=f'Diff: {diff_2m:.2f}', showarrow=True, arrowhead=1),
        dict(x='3M', y=vix_3_2, xref='x', yref='y', text=f'Diff: {diff_3m:.2f}', showarrow=True, arrowhead=1),
        dict(x='4M', y=vix_4_2, xref='x', yref='y', text=f'Diff: {diff_4m:.2f}', showarrow=True, arrowhead=1)
    ]

    # Create the layout for the plot
    layout = go.Layout(
        title=f'VIX Curves - {date1} and {date2}',
        xaxis=dict(title='Tenor'),
        yaxis=dict(title='Price'),
        annotations=annotations
    )

    # Create the figure and display the plot
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig
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
    long_contract = st.selectbox('Long Contract',('UX1', 'UX2', 'UX3'), index = 1)
    multiplier = st.number_input('Multiplier on long contract', value = 1.0, min_value = 0.0, max_value = 10.0)
    if_carry = st.checkbox('Carry')
    if if_carry:
        carry_threshold = st.number_input('Carry > threshold', value = 0.05, min_value = -1.0, max_value = 1.0)
    else:
        carry_threshold = False
    if_spread = st.checkbox('Spread')
    if if_spread:
        spread_threshold = st.number_input('Spread < threshold', value = 0.08, min_value = -1.00, max_value = 100.00)
    else:
        spread_threshold = False
    if_sl = st.checkbox('Stop Loss')
    if if_sl:
        stop_loss = st.number_input('Stop Loss Trigger', value = 37, min_value=0, max_value=200)
    else:
        stop_loss = False
    run = st.button('Run')

tab1, tab2, tab3 = st.tabs(['Single Strategy', 'Strategies Comparison', 'Term Structure'])
with tab1:
    if run:
        res_df = backtesting_vix_ls(short_contract=short_contract, long_contract=long_contract, multiplier=multiplier,
                                       start_date=start_date, end_date=end_date, carry_threshold=carry_threshold, 
                                       spread_threshold=spread_threshold, stop_loss=stop_loss)
        fig1 = plot_pnl(dfs = [res_df], short_contracts=[short_contract], 
               long_contracts=[long_contract],
               multipliers=[multiplier], 
               carry_thresholds=[carry_threshold], 
               spread_thresholds=[spread_threshold], 
               stop_losses=[stop_loss],
               start_date=start_date,end_date=end_date)
        
        perf_year = performance_summary_by_year(df = res_df)

        st.plotly_chart(fig1, use_container_width=True)

        # plot vix spot values
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        fig_vix = px.line(filtered_df, x = 'Date', y = 'VIX_Spot', title = 'VIX Spot')
        fig_vix.update_layout(
            xaxis=dict(
            zeroline=True,
            zerolinewidth=10,
            zerolinecolor='black',
            showgrid=True
            ))
        st.plotly_chart(fig_vix, use_container_width=True)

        st.dataframe(perf_year, use_container_width=True)


with tab2:
    if run:
        comp_df = pd.DataFrame()

        progress_text = 'Running backtesting.......'
        my_bar = st.progress(0, text=progress_text)
        percent_complete = 0

        res_df_1 = backtesting_vix_ls(short_contract=short_contract, long_contract=long_contract, multiplier=multiplier, start_date=start_date, end_date=end_date, carry_threshold=carry_threshold, spread_threshold=spread_threshold, stop_loss=stop_loss)
        comp_df['Pure_short_UX1'] = res_df_1['Cumulative_PnL']
        my_bar.progress(percent_complete + 20, text=progress_text)
        percent_complete += 20

        res_df_2 =  backtesting_vix_ls(short_contract=short_contract, long_contract=long_contract, multiplier=multiplier, start_date=start_date, end_date=end_date, carry_threshold=carry_threshold, spread_threshold=spread_threshold, stop_loss=stop_loss)
        comp_df['Short 1x Long 0.5x'] = res_df_2['Cumulative_PnL']
        my_bar.progress(percent_complete + 20, text=progress_text)
        percent_complete += 20
        
        res_df_3 =  backtesting_vix_ls(short_contract=short_contract, long_contract=long_contract, multiplier=multiplier, start_date=start_date, end_date=end_date, carry_threshold=carry_threshold, spread_threshold=spread_threshold, stop_loss=stop_loss)
        comp_df['Short 1x Long 1x'] = res_df_3['Cumulative_PnL']
        my_bar.progress(percent_complete + 20, text=progress_text)
        percent_complete += 20

        res_df_4 =  backtesting_vix_ls(short_contract=short_contract, long_contract=long_contract, multiplier=multiplier, start_date=start_date, end_date=end_date, carry_threshold=carry_threshold, spread_threshold=spread_threshold, stop_loss=stop_loss)
        comp_df['Short 1x Long 1.5x'] = res_df_4['Cumulative_PnL']
        my_bar.progress(percent_complete + 20, text=progress_text)
        percent_complete += 20

        res_df_5 =  backtesting_vix_ls(short_contract=short_contract, long_contract=long_contract, multiplier=multiplier, start_date=start_date, end_date=end_date, carry_threshold=carry_threshold, spread_threshold=spread_threshold, stop_loss=stop_loss)
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

with tab3:
    df3 = pd.read_csv('VIX_Price.csv')
    df3['Date'] = pd.to_datetime(df3['Date'], format='%d/%m/%Y')
    if run:
        term_structure_fig = plot_curves(date1=start_date, date2=end_date, df = df3)
        st.plotly_chart(term_structure_fig, use_container_width=True)
