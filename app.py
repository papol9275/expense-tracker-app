import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import calendar
import hashlib
import json
import shutil

# Set page configuration
st.set_page_config(page_title="Personal Finance Tracker", page_icon=":money_with_wings:", layout="wide")

# User data directory
USER_DATA_DIR = "user_data"
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

# User credentials file
CREDENTIALS_FILE = "user_credentials.json"
if not os.path.exists(CREDENTIALS_FILE):
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump({}, f)

# Global variables
DATA_FILE = "finance_data.csv"
BUDGET_FILE = "budget.csv"
EXPENSE_CATEGORIES = ["Housing", "Food", "Transportation", "Utilities", "Entertainment", 
              "Healthcare", "Personal Care", "Education", "Savings", "Debt Payments", "Other"]
INCOME_CATEGORIES = ["Salary", "Freelance", "Investment", "Gift", "Refund", "Business", "Other"]
CATEGORIES = EXPENSE_CATEGORIES  # For backward compatibility with existing data

# User Authentication Functions
def hash_password(password):
    """Create a SHA-256 hash of the password."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_user_credentials():
    """Load user credentials from JSON file."""
    try:
        with open(CREDENTIALS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_user_credentials(credentials):
    """Save user credentials to JSON file."""
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(credentials, f)

def register_user(username, password):
    """Register a new user."""
    credentials = load_user_credentials()
    
    if username in credentials:
        return False
    
    # Create user directory
    user_dir = os.path.join(USER_DATA_DIR, username)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    # Store hashed password
    credentials[username] = hash_password(password)
    save_user_credentials(credentials)
    
    # Initialize empty data files
    user_data_file = os.path.join(user_dir, DATA_FILE)
    user_budget_file = os.path.join(user_dir, BUDGET_FILE)
    
    if not os.path.exists(user_data_file):
        df = pd.DataFrame(columns=['Date', 'Category', 'Type', 'Amount', 'Description'])
        df.to_csv(user_data_file, index=False)
    
    if not os.path.exists(user_budget_file):
        budget_df = pd.DataFrame({'Category': CATEGORIES, 'Budget': np.zeros(len(CATEGORIES))})
        budget_df.to_csv(user_budget_file, index=False)
    
    return True

def authenticate_user(username, password):
    """Authenticate a user."""
    credentials = load_user_credentials()
    if username in credentials and credentials[username] == hash_password(password):
        return True
    return False

def get_user_data_path(filename):
    """Get the path to a user's data file."""
    if 'username' not in st.session_state:
        return filename
    
    return os.path.join(USER_DATA_DIR, st.session_state['username'], filename)

# Helper Functions
def load_data():
    """Load the finance data from CSV file or create a new dataframe if file doesn't exist."""
    user_data_file = get_user_data_path(DATA_FILE)
    
    if os.path.exists(user_data_file):
        df = pd.read_csv(user_data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    else:
        df = pd.DataFrame(columns=['Date', 'Category', 'Type', 'Amount', 'Description'])
        df['Date'] = pd.to_datetime(df['Date'])
        return df

def save_data(df):
    """Save the finance data to CSV file."""
    user_data_file = get_user_data_path(DATA_FILE)
    df.to_csv(user_data_file, index=False)

def add_transaction(date, category, transaction_type, amount, description=""):
    """Add a new transaction to the dataframe."""
    df = load_data()
    new_transaction = pd.DataFrame({
        'Date': [pd.to_datetime(date)],
        'Category': [category],
        'Type': [transaction_type],
        'Amount': [float(amount)],
        'Description': [description]
    })
    df = pd.concat([df, new_transaction], ignore_index=True)
    save_data(df)
    return df

def get_date_range(period):
    """Return start and end dates based on period selection."""
    today = datetime.now()
    if period == "This Week":
        start_date = today - timedelta(days=today.weekday())
        end_date = start_date + timedelta(days=6)  # Include full week
    elif period == "This Month":
        start_date = today.replace(day=1)
        # Get last day of current month
        if today.month == 12:
            end_date = today.replace(day=31)
        else:
            end_date = today.replace(month=today.month+1, day=1) - timedelta(days=1)
    elif period == "Last Month":
        last_month = today.replace(day=1) - timedelta(days=1)
        start_date = last_month.replace(day=1)
        end_date = last_month.replace(day=calendar.monthrange(last_month.year, last_month.month)[1])
    elif period == "Last 3 Months":
        start_date = (today - timedelta(days=90)).replace(day=1)
        end_date = today
    elif period == "This Year":
        start_date = today.replace(month=1, day=1)
        end_date = today.replace(month=12, day=31)  # Include full year
    else:  # All Time
        df = load_data()
        if df.empty:
            start_date = today
            end_date = today
        else:
            start_date = df['Date'].min()
            end_date = today
    
    return start_date, end_date

def filter_data(df, start_date, end_date, categories=None, transaction_types=None):
    """Filter data based on date range, categories, and transaction types."""
    filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & 
                     (df['Date'] <= pd.to_datetime(end_date))]
    
    if categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    if transaction_types:
        filtered_df = filtered_df[filtered_df['Type'].isin(transaction_types)]
    
    return filtered_df

def create_summary_metrics(df):
    """Create summary metrics for the dashboard."""
    if df.empty:
        return 0, 0, 0
    
    income = df[df['Type'] == 'Income']['Amount'].sum()
    expenses = df[df['Type'] == 'Expense']['Amount'].sum()
    balance = income - expenses
    
    return income, expenses, balance

def create_category_chart(df):
    """Create a bar chart of expenses by category."""
    if df.empty or df[df['Type'] == 'Expense'].empty:
        return None
    
    category_data = df[df['Type'] == 'Expense'].groupby('Category')['Amount'].sum().reset_index()
    category_data = category_data.sort_values('Amount', ascending=False)
    
    fig = px.bar(
        category_data,
        x='Category',
        y='Amount',
        title='Expenses by Category',
        color='Category',
        labels={'Amount': 'Amount (₱)', 'Category': 'Category'}
    )
    
    return fig

def create_time_series(df):
    """Create a time series chart of income vs expenses."""
    if df.empty:
        return None
    
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_data = df.groupby(['Month', 'Type'])['Amount'].sum().unstack().reset_index()
    monthly_data['Month'] = monthly_data['Month'].dt.to_timestamp()
    
    # Fill missing values with 0
    if 'Income' not in monthly_data.columns:
        monthly_data['Income'] = 0
    if 'Expense' not in monthly_data.columns:
        monthly_data['Expense'] = 0
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_data['Month'],
        y=monthly_data['Income'],
        name='Income',
        marker_color='#2E8B57'  # Green color for income
    ))
    fig.add_trace(go.Bar(
        x=monthly_data['Month'],
        y=monthly_data['Expense'],
        name='Expense',
        marker_color='#CD5C5C'  # Red color for expense
    ))
    
    fig.update_layout(
        title='Monthly Income vs Expenses',
        xaxis_title='Month',
        yaxis_title='Amount (₱)',
        legend_title='Type',
        barmode='group'  # Group bars side by side
    )
    
    return fig

def load_budget():
    """Load budget data from CSV file or create a new one if it doesn't exist."""
    user_budget_file = get_user_data_path(BUDGET_FILE)
    
    if os.path.exists(user_budget_file):
        budget_df = pd.read_csv(user_budget_file)
        return budget_df
    else:
        budget_df = pd.DataFrame({'Category': CATEGORIES, 'Budget': np.zeros(len(CATEGORIES))})
        budget_df.to_csv(user_budget_file, index=False)
        return budget_df

def save_budget(budget_df):
    """Save the budget data to CSV file."""
    user_budget_file = get_user_data_path(BUDGET_FILE)
    budget_df.to_csv(user_budget_file, index=False)

def compare_with_budget(df, period):
    """Compare expenses with budget for the current month."""
    budget_df = load_budget()
    
    start_date, end_date = get_date_range(period)
    filtered_df = filter_data(df, start_date, end_date, transaction_types=['Expense'])
    
    if filtered_df.empty:
        return None
    
    category_expenses = filtered_df.groupby('Category')['Amount'].sum().reset_index()
    
    # Merge with budget data
    comparison = pd.merge(category_expenses, budget_df, on='Category', how='left')
    comparison['Budget'].fillna(0, inplace=True)
    comparison['Remaining'] = comparison['Budget'] - comparison['Amount']
    comparison['Percentage'] = (comparison['Amount'] / comparison['Budget'] * 100).round(1)
    comparison['Percentage'].replace([np.inf, -np.inf], 0, inplace=True)
    comparison['Percentage'].fillna(0, inplace=True)
    
    return comparison

def migrate_existing_data():
    """Migrate existing data files to the user's account if they exist."""
    if 'username' not in st.session_state:
        return False
        
    username = st.session_state['username']
    user_dir = os.path.join(USER_DATA_DIR, username)
    
    migrated = False
    
    # Check for global data file and migrate if exists
    if os.path.exists(DATA_FILE):
        try:
            data_df = pd.read_csv(DATA_FILE)
            user_data_file = os.path.join(user_dir, DATA_FILE)
            data_df.to_csv(user_data_file, index=False)
            migrated = True
        except Exception as e:
            st.error(f"Error migrating transaction data: {str(e)}")
    
    # Check for global budget file and migrate if exists
    if os.path.exists(BUDGET_FILE):
        try:
            budget_df = pd.read_csv(BUDGET_FILE)
            user_budget_file = os.path.join(user_dir, BUDGET_FILE)
            budget_df.to_csv(user_budget_file, index=False)
            migrated = True
        except Exception as e:
            st.error(f"Error migrating budget data: {str(e)}")
            
    return migrated

# Main App Layout
def main():
    # Initialize session state variables for authentication
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'show_migration' not in st.session_state:
        st.session_state['show_migration'] = False
    
    # Check if user is authenticated
    if not st.session_state['authenticated']:
        display_login()
    else:
        # Handle data migration if needed
        if st.session_state['show_migration']:
            st.title("Import Existing Data")
            st.write(f"Welcome, {st.session_state['username']}! We've detected existing data.")
            st.write("Would you like to import this data into your account?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, import data"):
                    migrated = migrate_existing_data()
                    if migrated:
                        st.success("Data successfully imported to your account!")
                    else:
                        st.info("No data was imported.")
                    st.session_state['show_migration'] = False
                    st.rerun()
            with col2:
                if st.button("No, start fresh"):
                    st.session_state['show_migration'] = False
                    st.rerun()
            
            # Exit here to ensure migration decision is made first
            return
        
        # Load data for the authenticated user
        df = load_data()
        
        # Add logout button to the sidebar
        if st.sidebar.button("Logout"):
            st.session_state['authenticated'] = False
            st.session_state['username'] = None
            st.rerun()
        
        # Show username in sidebar
        st.sidebar.write(f"Logged in as: **{st.session_state['username']}**")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        
        # Creating clickable navigation options
        page = st.sidebar.radio(
            "Go to",
            ["Dashboard", "Add Transaction", "View/Edit Transactions", "Reports", "Budget Management"],
            label_visibility="collapsed"
        )
        
        if page == "Dashboard":
            display_dashboard(df)
        elif page == "Add Transaction":
            display_add_transaction()
        elif page == "View/Edit Transactions":
            display_view_edit(df)
        elif page == "Reports":
            display_reports(df)
        elif page == "Budget Management":
            display_budget_management()

def display_dashboard(df):
    st.title("Finance Tracker Dashboard")
    
    # Time period filter
    period = st.selectbox("Period", ["This Week", "This Month", "Last Month", "Last 3 Months", "This Year", "All Time"])
    start_date, end_date = get_date_range(period)
    
    # Filter data
    filtered_df = filter_data(df, start_date, end_date)
    
    # Summary metrics
    st.subheader("Summary")
    income, expenses, balance = create_summary_metrics(filtered_df)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Income", f"₱{income:,.2f}")
    col2.metric("Expenses", f"₱{expenses:,.2f}")
    col3.metric("Balance", f"₱{balance:,.2f}", delta=f"{balance:,.2f}")
    
    # Category breakdown
    st.subheader("Expense Breakdown")
    category_chart = create_category_chart(filtered_df)
    if category_chart:
        st.plotly_chart(category_chart, use_container_width=True)
    else:
        st.info("No expense data available for the selected period.")
    
    # Recent transactions
    st.subheader("Recent Transactions")
    if not filtered_df.empty:
        recent_transactions = filtered_df.sort_values('Date', ascending=False).head(5)
        st.dataframe(
            recent_transactions[['Date', 'Category', 'Type', 'Amount', 'Description']],
            use_container_width=True,
            column_config={
                "Date": st.column_config.DateColumn("Date"),
                "Amount": st.column_config.NumberColumn("Amount")
            }
        )
    else:
        st.info("No transactions available for the selected period.")
    
    # Quick add expense
    st.subheader("Quick Add Expense")
    with st.expander("Add a new expense quickly"):
        with st.form("quick_expense_form"):
            col1, col2 = st.columns(2)
            quick_date = col1.date_input("Date", value=datetime.now(), max_value=datetime(2050, 12, 31))
            quick_category = col2.selectbox("Category", CATEGORIES)
            quick_amount = col1.number_input("Amount (₱)", min_value=0.01, step=0.01)
            quick_description = col2.text_input("Description (optional)")
            
            quick_submit = st.form_submit_button("Add Expense")
            
            if quick_submit:
                if quick_amount <= 0:
                    st.error("Amount must be greater than zero.")
                else:
                    add_transaction(quick_date, quick_category, "Expense", quick_amount, quick_description)
                    st.success("Expense added successfully!")
                    st.rerun()

def display_add_transaction():
    st.title("Add Transaction")
    
    # Initialize session state for transaction type if it doesn't exist
    if 'transaction_type' not in st.session_state:
        st.session_state.transaction_type = "Expense"
    
    # Transaction type selection outside the form
    transaction_type_outside = st.selectbox(
        "Transaction Type", 
        ["Expense", "Income"],
        index=0 if st.session_state.transaction_type == "Expense" else 1,
        key="transaction_type_selector"
    )
    
    # Update session state based on selection
    st.session_state.transaction_type = transaction_type_outside
    
    # Select appropriate categories based on transaction type
    categories_to_show = EXPENSE_CATEGORIES if st.session_state.transaction_type == "Expense" else INCOME_CATEGORIES
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        transaction_date = col1.date_input("Date", value=datetime.now(), max_value=datetime(2050, 12, 31))
        # Hidden field to store the transaction type inside the form
        transaction_type = st.session_state.transaction_type
        
        transaction_category = col1.selectbox("Category", categories_to_show)
        transaction_amount = col2.number_input("Amount (₱)", min_value=0.01, step=0.01)
        
        transaction_description = st.text_area("Description (optional)")
        
        submit_button = st.form_submit_button("Add Transaction")
        
        if submit_button:
            if transaction_amount <= 0:
                st.error("Amount must be greater than zero.")
            else:
                add_transaction(transaction_date, transaction_category, transaction_type, 
                              transaction_amount, transaction_description)
                st.success(f"{transaction_type} added successfully!")

def display_view_edit(df):
    st.title("View/Edit Transactions")
    
    # Initialize session state for transaction editing if not exists
    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = False
    if 'transaction_to_edit' not in st.session_state:
        st.session_state.transaction_to_edit = None
    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = False
    if 'transaction_to_delete' not in st.session_state:
        st.session_state.transaction_to_delete = None
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    with col3:
        filter_type = st.multiselect("Transaction Type", ["Expense", "Income"], default=["Expense", "Income"])
    
    filter_category = st.multiselect("Categories", EXPENSE_CATEGORIES + INCOME_CATEGORIES, default=[])
    search_term = st.text_input("Search in description", "")
    
    # Apply filters
    filtered_df = filter_data(df, start_date, end_date, filter_category, filter_type)
    
    # Search in description
    if search_term:
        filtered_df = filtered_df[filtered_df['Description'].str.contains(search_term, case=False, na=False)]
    
    # Display filtered transactions
    if not filtered_df.empty:
        st.subheader(f"Showing {len(filtered_df)} transactions")
        
        # Create a dataframe for display with action buttons
        display_df = filtered_df.copy()
        
        # Reset index for proper reference
        display_df = display_df.reset_index(drop=True)
        
        # Function to handle edit button click
        def handle_edit(idx):
            st.session_state.edit_mode = True
            st.session_state.transaction_to_edit = idx
            
        # Function to handle delete button click
        def handle_delete(idx):
            st.session_state.confirm_delete = True
            st.session_state.transaction_to_delete = idx
        
        # Use an interactive dataframe with buttons
        with st.container():
            # Ensure data types are correct before passing to data_editor
            display_copy = display_df.copy()
            display_copy['Description'] = display_copy['Description'].astype(str)  # Convert Description to string
            
            # Show the data with edit options using a custom dataframe
            edited_df = st.data_editor(
                display_copy[['Date', 'Category', 'Type', 'Amount', 'Description']],
                use_container_width=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date"),
                    "Category": st.column_config.SelectboxColumn(
                        "Category", 
                        options=EXPENSE_CATEGORIES + INCOME_CATEGORIES
                    ),
                    "Type": st.column_config.SelectboxColumn(
                        "Type", 
                        options=["Expense", "Income"]
                    ),
                    "Amount": st.column_config.NumberColumn("Amount", format="₱%.2f"),
                    "Description": st.column_config.TextColumn("Description")
                },
                num_rows="dynamic",
                key="transaction_editor"
            )
            
            # Check if any changes were made in the data editor
            if not edited_df.equals(display_copy[['Date', 'Category', 'Type', 'Amount', 'Description']]):
                # Track if any changes were made
                changes_made = False
                
                # Find the indices in the original dataframe for updates
                for i, row in edited_df.iterrows():
                    # Check if this is an existing row or a new row
                    is_new_row = i >= len(display_df)
                    
                    if is_new_row:
                        # This is a new row, add it as a new transaction
                        if pd.notna(row['Date']) and pd.notna(row['Category']) and pd.notna(row['Type']) and pd.notna(row['Amount']):
                            # Ensure Description is a string and handle NaN values
                            description = str(row['Description']) if pd.notna(row['Description']) else ""
                            
                            # Add the new transaction
                            add_transaction(
                                date=row['Date'],
                                category=row['Category'],
                                transaction_type=row['Type'],
                                amount=float(row['Amount']),
                                description=description
                            )
                            changes_made = True
                    else:
                        # This is an existing row, check if it was modified
                        orig_row = display_df.iloc[i]
                        
                        # Check if this row was modified
                        if (row['Date'] != orig_row['Date'] or 
                            row['Category'] != orig_row['Category'] or
                            row['Type'] != orig_row['Type'] or
                            row['Amount'] != orig_row['Amount'] or
                            str(row['Description']) != str(orig_row['Description'])):
                            
                            # Find the index in the original dataframe
                            orig_idx = df.index[
                                (df['Date'] == orig_row['Date']) & 
                                (df['Category'] == orig_row['Category']) & 
                                (df['Type'] == orig_row['Type']) & 
                                (df['Amount'] == orig_row['Amount']) & 
                                (df['Description'].astype(str) == str(orig_row['Description']))
                            ].tolist()
                            
                            if orig_idx:
                                # Update the original dataframe
                                df.at[orig_idx[0], 'Date'] = pd.to_datetime(row['Date'])
                                df.at[orig_idx[0], 'Category'] = row['Category']
                                df.at[orig_idx[0], 'Type'] = row['Type']
                                df.at[orig_idx[0], 'Amount'] = float(row['Amount'])
                                df.at[orig_idx[0], 'Description'] = str(row['Description'])
                                changes_made = True
                
                if changes_made:
                    # Save the updated dataframe
                    save_data(df)
                    st.success("Transactions updated successfully!")
                    st.rerun()
        
        # Add buttons for delete functionality
        st.subheader("Delete Transactions")
        
        # Create a selectbox for easier transaction selection
        st.write("Select a transaction to delete:")
        transaction_options = [f"{row['Date'].strftime('%Y-%m-%d')} | {row['Category']} | {row['Type']} | ₱{row['Amount']:,.2f} | {str(row['Description'])[:30]}" 
                            for idx, row in display_df.iterrows()]
        
        selected_idx = st.selectbox(
            "Select transaction:",
            range(len(transaction_options)),
            format_func=lambda i: transaction_options[i],
            label_visibility="collapsed"
        )
        
        # Create columns for delete button and transaction details
        col1, col2 = st.columns([1, 4])
        with col1:
            delete_pressed = st.button("Delete Transaction", type="primary", use_container_width=True)
        with col2:
            if selected_idx is not None:
                selected_row = display_df.iloc[selected_idx]
                st.info(f"Selected: {selected_row['Date'].strftime('%Y-%m-%d')} - {selected_row['Category']} ({selected_row['Type']}) - ₱{selected_row['Amount']:,.2f}")
        
        # Handle delete confirmation
        if delete_pressed:
            st.session_state.confirm_delete = True
            st.session_state.transaction_to_delete = selected_idx
            
        if st.session_state.confirm_delete:
            st.warning("Are you sure you want to delete this transaction? This action cannot be undone.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete"):
                    # Get the transaction from the displayed dataframe
                    transaction_to_delete = display_df.iloc[st.session_state.transaction_to_delete]
                    
                    # Find the matching row in the original dataframe
                    match_idx = df.index[
                        (df['Date'] == transaction_to_delete['Date']) &
                        (df['Category'] == transaction_to_delete['Category']) &
                        (df['Type'] == transaction_to_delete['Type']) &
                        (df['Amount'] == transaction_to_delete['Amount']) &
                        (df['Description'] == transaction_to_delete['Description'])
                    ].tolist()
                    
                    if match_idx:
                        # Delete the transaction from the original dataframe
                        df = df.drop(match_idx[0]).reset_index(drop=True)
                        
                        # Save the updated dataframe
                        save_data(df)
                        st.success("Transaction deleted successfully!")
                        
                        # Reset the state
                        st.session_state.confirm_delete = False
                        st.session_state.transaction_to_delete = None
                        st.rerun()
                    else:
                        st.error("Transaction could not be found in the database.")
            
            with col2:
                if st.button("Cancel"):
                    # Reset the state
                    st.session_state.confirm_delete = False
                    st.session_state.transaction_to_delete = None
                    st.rerun()
    else:
        st.info("No transactions found with the current filters.")

def display_reports(df):
    st.title("Financial Reports")
    
    # Time period selection
    report_period = st.selectbox("Select Period", ["This Month", "Last Month", "Last 3 Months", "This Year", "All Time"])
    start_date, end_date = get_date_range(report_period)
    
    # Filter data
    filtered_df = filter_data(df, start_date, end_date)
    
    if filtered_df.empty:
        st.info("No data available for the selected period.")
        return
    
    # Income vs Expenses
    st.subheader("Income vs Expenses")
    income_expense_chart = create_time_series(filtered_df)
    if income_expense_chart:
        st.plotly_chart(income_expense_chart, use_container_width=True)
    
    # Category breakdown for the period
    st.subheader("Expense Categories")
    category_chart = create_category_chart(filtered_df)
    if category_chart:
        st.plotly_chart(category_chart, use_container_width=True)
    
    # Budget comparison
    st.subheader("Budget Comparison")
    budget_comparison = compare_with_budget(df, report_period)
    if budget_comparison is not None and not budget_comparison.empty:
        # Create a progress bar for each category
        for _, row in budget_comparison.iterrows():
            if row['Budget'] > 0:  # Only show categories with a budget
                col1, col2 = st.columns([3, 1])
                with col1:
                    progress = min(100, row['Percentage'])
                    progress_color = 'normal' if progress < 85 else 'warning' if progress < 100 else 'error'
                    st.progress(progress / 100, text=f"{row['Category']} - ₱{row['Amount']:,.2f} of ₱{row['Budget']:,.2f} ({progress}%)")
                with col2:
                    st.write(f"Remaining: ₱{row['Remaining']:,.2f}")
    else:
        st.info("No budget data available or no expenses in the selected period.")
    
    # Transaction trend by day of week
    if len(filtered_df) > 5:  # Only show if we have enough data
        st.subheader("Spending by Day of Week")
        filtered_df['DayOfWeek'] = filtered_df['Date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        expenses_by_day = filtered_df[filtered_df['Type'] == 'Expense'].groupby('DayOfWeek')['Amount'].sum().reindex(day_order).reset_index()
        
        fig = px.bar(
            expenses_by_day,
            x='DayOfWeek',
            y='Amount',
            title='Expenses by Day of Week',
            labels={'Amount': 'Total Expenses (₱)', 'DayOfWeek': 'Day of Week'},
            color='DayOfWeek'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_budget_management():
    st.title("Budget Management")
    
    # Load budget data
    budget_df = load_budget()
    
    # Display current month's budget vs. actual
    st.subheader("This Month's Budget Status")
    df = load_data()
    budget_status = compare_with_budget(df, "This Month")
    
    if budget_status is not None and not budget_status.empty:
        # Create a progress bar for each category
        for _, row in budget_status.iterrows():
            if row['Budget'] > 0:  # Only show categories with a budget
                col1, col2 = st.columns([3, 1])
                with col1:
                    progress = min(100, row['Percentage'])
                    progress_color = 'normal' if progress < 85 else 'warning' if progress < 100 else 'error'
                    st.progress(progress / 100, text=f"{row['Category']} - ₱{row['Amount']:,.2f} of ₱{row['Budget']:,.2f} ({progress}%)")
                with col2:
                    st.write(f"Remaining: ₱{row['Remaining']:,.2f}")
    
    # Set budget form
    st.subheader("Set Monthly Budget")
    with st.form("budget_form"):
        for i, category in enumerate(CATEGORIES):
            current_budget = budget_df.loc[budget_df['Category'] == category, 'Budget'].values[0] if not budget_df.empty else 0
            col1, col2 = st.columns([1, 3])
            col1.write(category)
            budget_value = col2.number_input(f"Budget for {category} (₱)", 
                                            min_value=0.0, 
                                            value=float(current_budget), 
                                            step=10.0,
                                            key=f"budget_{i}",
                                            label_visibility="collapsed")
            budget_df.loc[budget_df['Category'] == category, 'Budget'] = budget_value
        
        if st.form_submit_button("Save Budget"):
            save_budget(budget_df)
            st.success("Budget updated successfully!")

# Login and Registration UI
def display_login():
    st.title("Personal Finance Tracker - Login")
    
    # Tabs for Login and Register
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        # Login Form
        st.subheader("Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if authenticate_user(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.success(f"Welcome back, {username}!")
                    
                    # Check if this is the first login (no data files exist)
                    user_dir = os.path.join(USER_DATA_DIR, username)
                    user_data_file = os.path.join(user_dir, DATA_FILE)
                    
                    if not os.path.exists(user_data_file) and (os.path.exists(DATA_FILE) or os.path.exists(BUDGET_FILE)):
                        st.session_state['show_migration'] = True
                    
                    st.rerun()
                else:
                    st.error("Invalid username or password!")
    
    with tab2:
        # Registration Form
        st.subheader("Create New Account")
        with st.form("register_form"):
            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_button = st.form_submit_button("Register")
            
            if submit_button:
                if not new_username or len(new_username) < 3:
                    st.error("Username must be at least 3 characters long.")
                elif not new_password or len(new_password) < 6:
                    st.error("Password must be at least 6 characters long.")
                elif new_password != confirm_password:
                    st.error("Passwords don't match!")
                else:
                    if register_user(new_username, new_password):
                        st.success("Registration successful! You can now log in.")
                        
                        # Offer data migration for new users if legacy data exists
                        if os.path.exists(DATA_FILE) or os.path.exists(BUDGET_FILE):
                            st.session_state['show_migration'] = True
                            st.session_state['authenticated'] = True
                            st.session_state['username'] = new_username
                            st.rerun()
                    else:
                        st.error("Username already taken. Please choose another one.")

if __name__ == "__main__":
    main()
