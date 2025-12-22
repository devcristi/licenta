import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from collections import defaultdict

# Load the JSON file
json_file = r"D:\study\licenta\creier\dataset\BRATS\brats_metadata_splits.json"

print("Loading data...")
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare data for analysis
splits_data = defaultdict(lambda: {'patients': set(), 'visits': 0})
visits_per_patient = defaultdict(lambda: defaultdict(int))
file_types = defaultdict(int)

for entry in data:
    patient_id = entry['patient_id']
    split = entry['final_split']
    
    splits_data[split]['patients'].add(patient_id)
    splits_data[split]['visits'] += 1
    visits_per_patient[split][int(entry['visit_id'])] += 1
    
    # Count file types
    for file_obj in entry['all_files']:
        file_type = file_obj.get('type', 'unknown')
        file_types[file_type] += 1

# Create summary dataframe
summary_data = {
    'Split': [],
    'Patients': [],
    'Visits': [],
    'Avg Visits': []
}

for split in ['train', 'val_from_train', 'test']:
    num_patients = len(splits_data[split]['patients'])
    num_visits = splits_data[split]['visits']
    avg_visits = num_visits / num_patients if num_patients > 0 else 0
    
    summary_data['Split'].append(split.replace('_', ' ').title())
    summary_data['Patients'].append(num_patients)
    summary_data['Visits'].append(num_visits)
    summary_data['Avg Visits'].append(avg_visits)

df_summary = pd.DataFrame(summary_data)

# Figure 1: Dataset Split Overview (Pie chart)
fig1 = go.Figure(data=[
    go.Pie(
        labels=df_summary['Split'],
        values=df_summary['Patients'],
        hole=0.3,
        textposition='inside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Patients: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
])

fig1.update_layout(
    title={
        'text': '<b>Dataset Split: Patient Distribution</b>',
        'font': {'size': 20}
    },
    height=600,
    showlegend=True,
    hovermode='closest'
)
fig1.write_html(r"D:\study\licenta\creier\scripts\plots\01_split_distribution.html")
print("✅ 01_split_distribution.html")

# Figure 2: Visits per Split (Bar chart)
fig2 = go.Figure(data=[
    go.Bar(
        x=df_summary['Split'],
        y=df_summary['Visits'],
        text=df_summary['Visits'],
        textposition='outside',
        marker=dict(
            color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{x}</b><br>Visits: %{y}<extra></extra>'
    )
])

fig2.update_layout(
    title='<b>Total Visits per Split</b>',
    xaxis_title='Split',
    yaxis_title='Number of Visits',
    height=500,
    showlegend=False,
    hovermode='x unified'
)
fig2.write_html(r"D:\study\licenta\creier\scripts\plots\02_visits_per_split.html")
print("✅ 02_visits_per_split.html")

# Figure 3: Average Visits per Patient
fig3 = go.Figure(data=[
    go.Bar(
        x=df_summary['Split'],
        y=df_summary['Avg Visits'],
        text=[f'{x:.2f}' for x in df_summary['Avg Visits']],
        textposition='outside',
        marker=dict(
            color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{x}</b><br>Avg Visits: %{y:.2f}<extra></extra>'
    )
])

fig3.update_layout(
    title='<b>Average Visits per Patient by Split</b>',
    xaxis_title='Split',
    yaxis_title='Average Visits',
    height=500,
    showlegend=False,
    hovermode='x unified'
)
fig3.write_html(r"D:\study\licenta\creier\scripts\plots\03_avg_visits_per_patient.html")
print("✅ 03_avg_visits_per_patient.html")

# Figure 4: Patients vs Visits Comparison
fig4 = go.Figure()

fig4.add_trace(go.Bar(
    x=df_summary['Split'],
    y=df_summary['Patients'],
    name='Patients',
    marker_color='#1f77b4'
))

fig4.add_trace(go.Bar(
    x=df_summary['Split'],
    y=df_summary['Visits'],
    name='Visits',
    marker_color='#ff7f0e'
))

fig4.update_layout(
    title='<b>Patients vs Visits Comparison</b>',
    xaxis_title='Split',
    yaxis_title='Count',
    barmode='group',
    height=500,
    hovermode='x unified'
)
fig4.write_html(r"D:\study\licenta\creier\scripts\plots\04_patients_vs_visits.html")
print("✅ 04_patients_vs_visits.html")

# Figure 5: MRI File Types Distribution
file_types_df = pd.DataFrame([
    {'Type': 'T1 Native', 'Count': file_types.get('t1_native', 0)},
    {'Type': 'T1 Contrast', 'Count': file_types.get('t1_contrast', 0)},
    {'Type': 'T2', 'Count': file_types.get('t2', 0)},
    {'Type': 'T2w', 'Count': file_types.get('t2w', 0)},
    {'Type': 'FLAIR', 'Count': file_types.get('flair', 0)},
    {'Type': 'Segmentation', 'Count': file_types.get('segmentation', 0)},
])

fig5 = go.Figure(data=[
    go.Bar(
        x=file_types_df['Type'],
        y=file_types_df['Count'],
        text=file_types_df['Count'],
        textposition='outside',
        marker=dict(
            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        ),
        hovertemplate='<b>%{x}</b><br>Files: %{y}<extra></extra>'
    )
])

fig5.update_layout(
    title='<b>MRI File Types Distribution</b>',
    xaxis_title='File Type',
    yaxis_title='Number of Files',
    height=500,
    showlegend=False,
    xaxis_tickangle=-45,
    hovermode='x unified'
)
fig5.write_html(r"D:\study\licenta\creier\scripts\plots\05_mri_file_types.html")
print("✅ 05_mri_file_types.html")

# Figure 6: Summary Statistics Table
fig6 = go.Figure(data=[go.Table(
    header=dict(
        values=['<b>Metric</b>', '<b>Train (80%)</b>', '<b>Validation (20%)</b>', '<b>Test</b>'],
        fill_color='#1f77b4',
        font=dict(color='white', size=12),
        align='center'
    ),
    cells=dict(
        values=[
            ['Patients', 'Visits', 'Avg Visits/Patient'],
            [
                f"{df_summary[df_summary['Split']=='Train']['Patients'].values[0]:.0f}",
                f"{df_summary[df_summary['Split']=='Train']['Visits'].values[0]:.0f}",
                f"{df_summary[df_summary['Split']=='Train']['Avg Visits'].values[0]:.2f}"
            ],
            [
                f"{df_summary[df_summary['Split']=='Val From Train']['Patients'].values[0]:.0f}",
                f"{df_summary[df_summary['Split']=='Val From Train']['Visits'].values[0]:.0f}",
                f"{df_summary[df_summary['Split']=='Val From Train']['Avg Visits'].values[0]:.2f}"
            ],
            [
                f"{df_summary[df_summary['Split']=='Test']['Patients'].values[0]:.0f}",
                f"{df_summary[df_summary['Split']=='Test']['Visits'].values[0]:.0f}",
                f"{df_summary[df_summary['Split']=='Test']['Avg Visits'].values[0]:.2f}"
            ]
        ],
        fill_color='#f0f0f0',
        font=dict(size=11),
        align='center',
        height=30
    )
)
])

fig6.update_layout(
    title='<b>Dataset Statistics Summary</b>',
    height=400,
    showlegend=False
)
fig6.write_html(r"D:\study\licenta\creier\scripts\plots\06_statistics_table.html")
print("✅ 06_statistics_table.html")

# Figure 7: Donut chart - Data Distribution
labels = ['Train', 'Validation', 'Test']
values = df_summary['Visits'].tolist()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig7 = go.Figure(data=[
    go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textposition='inside',
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>Visits: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
])

fig7.update_layout(
    title='<b>Visit Distribution Across Splits</b>',
    height=600,
    showlegend=True,
    hovermode='closest'
)
fig7.write_html(r"D:\study\licenta\creier\scripts\plots\07_visit_distribution_donut.html")
print("✅ 07_visit_distribution_donut.html")

# Create index HTML
index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>BRATS Dataset Statistics</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1f77b4;
            text-align: center;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .card {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .card:hover {
            box-shadow: 0 4px 12px rgba(31,119,180,0.3);
            transform: translateY(-2px);
        }
        .card h3 {
            color: #1f77b4;
            margin-top: 0;
        }
        .card a {
            color: #1f77b4;
            text-decoration: none;
            font-weight: bold;
        }
        .card a:hover {
            text-decoration: underline;
        }
        .summary {
            background-color: #e8f4f8;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 BRATS Dataset Statistics Dashboard</h1>
        
        <div class="summary">
            <h3>Dataset Overview</h3>
            <p><strong>Total Patients:</strong> 818 | <strong>Total Visits:</strong> 1,809</p>
            <p><strong>Train:</strong> 584 patients (1,324 visits) | 
               <strong>Validation:</strong> 147 patients (297 visits) | 
               <strong>Test:</strong> 87 patients (188 visits)</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📊 Split Distribution</h3>
                <p>Patient distribution across train/validation/test splits</p>
                <a href="01_split_distribution.html">View Chart →</a>
            </div>
            
            <div class="card">
                <h3>📈 Visits per Split</h3>
                <p>Total number of MRI visits per split</p>
                <a href="02_visits_per_split.html">View Chart →</a>
            </div>
            
            <div class="card">
                <h3>📉 Average Visits</h3>
                <p>Average number of visits per patient in each split</p>
                <a href="03_avg_visits_per_patient.html">View Chart →</a>
            </div>
            
            <div class="card">
                <h3>🔄 Patients vs Visits</h3>
                <p>Comparison between patient count and visit count</p>
                <a href="04_patients_vs_visits.html">View Chart →</a>
            </div>
            
            <div class="card">
                <h3>🏥 MRI File Types</h3>
                <p>Distribution of MRI modalities (T1, T2, FLAIR, Segmentation)</p>
                <a href="05_mri_file_types.html">View Chart →</a>
            </div>
            
            <div class="card">
                <h3>📋 Statistics Table</h3>
                <p>Detailed summary statistics for each split</p>
                <a href="06_statistics_table.html">View Table →</a>
            </div>
            
            <div class="card">
                <h3>🍩 Visit Distribution</h3>
                <p>Donut chart showing visit percentages</p>
                <a href="07_visit_distribution_donut.html">View Chart →</a>
            </div>
        </div>
    </div>
</body>
</html>
"""

with open(r"D:\study\licenta\creier\scripts\plots\index.html", 'w', encoding='utf-8') as f:
    f.write(index_html)

print("✅ index.html - Dashboard")
print("\n" + "="*60)
print("✅ All visualizations created successfully!")
print("="*60)
print(f"\nOpen: D:\\study\\licenta\\creier\\scripts\\plots\\index.html")
