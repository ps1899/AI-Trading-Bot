# AI-Trading-Bot

A **Trading Bot** is an AI powered program which implements the deployed trading algorithms according to the user/developer. 
It is a part of a fully implemented Algorithmic Trading System which buys, sell or hold off the stock which ever deems profitable.

Here, I have developed an AI powered trading bot which will use trading signals for buying, selling or holding off the stocks. The system will trade for a whole day and 
then again ask for more data for trading from the stock_data.csv. It will store the data in a hierarchical data format file (HDF5) for the current week and the last week.
The infinite loop (threaded for concurrent systems) is responsible for gathering data once a day, and determining whether or not we have reached a weekly split yet. Upon reaching 
a weekly split, the variables are updated and we consult our AI on whether or not to buy or to sell.

I have deployed/hosted the bot on the **Google Cloud Platform** to mitigate the complications of running it 24/7 onto the computer systems.

P.S: Please don't try to use the algorithm for live trading, not liable for any loss associated with this algorithm.
