from flask import Flask, render_template, jsonify, request
import requests
import os
import json
import numpy as np
from datetime import datetime, timedelta
import logging
import time

# --------------------------
# Flask App Setup
# --------------------------
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# Supabase Configuration
# --------------------------
SUPABASE_URL = 'https://eqyitfzewdglvoqpmxqu.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVxeWl0Znpld2RnbHZvcXBteHF1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQyNTYyMTYsImV4cCI6MjA3OTgzMjIxNn0.jal1hwOu5y_bBHDTbj-fvmCZS2DmlgJIxalPz-qCxKI'
SUPABASE_TABLE = 'sensor_data'

headers = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'Content-Type': 'application/json'
}

# Custom caching decorator with TTL (PythonAnywhere has Python 3.9, not Python 3.10+)
def cache_with_ttl(ttl_seconds=30):
    """Custom cache decorator with TTL support for Python 3.9"""
    def decorator(func):
        cache = {}
        cache_timestamps = {}

        def wrapper(*args, **kwargs):
            # Create a key based on function arguments
            key = str(args) + str(kwargs)

            current_time = time.time()

            # Check if cached result exists and is still valid
            if key in cache:
                if current_time - cache_timestamps[key] < ttl_seconds:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache[key]
                else:
                    logger.debug(f"Cache expired for {func.__name__}")

            # Call function and cache result
            result = func(*args, **kwargs)
            cache[key] = result
            cache_timestamps[key] = current_time

            return result

        return wrapper

    return decorator

# --------------------------
# Enhanced Helper Functions
# --------------------------
def safe_float(value):
    """Safely convert value to float, handling None and invalid values"""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def safe_int(value):
    """Safely convert value to int, handling None and invalid values"""
    if value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

def calculate_health_status(health_score):
    """Convert health score to status"""
    health_score = safe_float(health_score)
    if health_score >= 90:
        return "EXCELLENT", "success"
    elif health_score >= 80:
        return "GOOD", "success"
    elif health_score >= 70:
        return "FAIR", "warning"
    elif health_score >= 60:
        return "CONCERN", "warning"
    elif health_score >= 40:
        return "POOR", "danger"
    else:
        return "CRITICAL", "danger"

def get_trend_icon(trend_value):
    """Get trend icon based on value"""
    trend_value = safe_float(trend_value)
    if trend_value > 0.5:
        return "↑", "up", "Rising rapidly"
    elif trend_value > 0.1:
        return "↗", "up", "Rising"
    elif trend_value < -0.5:
        return "↓", "down", "Falling rapidly"
    elif trend_value < -0.1:
        return "↘", "down", "Falling"
    else:
        return "→", "stable", "Stable"

def calculate_enhanced_statistics(data):
    """Calculate enhanced statistics from sensor data"""
    if not data:
        return {
            'total_records': 0,
            'avg_health_score': 0,
            'avg_data_quality': 0,
            'alerts_count': 0,
            'alerts_list': [],
            'cow_analysis': {},
            'system_status': 'no_data'
        }

    # Group by cow
    cow_stats = {}
    alerts = []

    for item in data:
        cow_id = item.get('cow_id', 'Unknown')

        # Initialize cow stats
        if cow_id not in cow_stats:
            cow_stats[cow_id] = {
                'temps': [], 'bpms': [], 'spo2s': [],
                'health_scores': [], 'data_qualities': [],
                'health_statuses': [], 'timestamps': []
            }

        # Collect data with safe conversion
        stats = cow_stats[cow_id]
        temp = safe_float(item.get('temperature'))
        bpm = safe_int(item.get('bpm'))
        spo2 = safe_int(item.get('spo2'))
        health_score = safe_float(item.get('health_score'))
        data_quality = safe_float(item.get('data_quality'))

        if temp is not None:
            stats['temps'].append(temp)
        if bpm is not None:
            stats['bpms'].append(bpm)
        if spo2 is not None:
            stats['spo2s'].append(spo2)
        if health_score is not None:
            stats['health_scores'].append(health_score)
        if data_quality is not None:
            stats['data_qualities'].append(data_quality)
        if item.get('health_status') is not None:
            stats['health_statuses'].append(item['health_status'])

        stats['timestamps'].append(item.get('created_at'))

        # Generate alerts based on enhanced data
        health_status = item.get('health_status', 'NORMAL')

        # Enhanced temperature alerts
        if temp > 40.0:
            alerts.append({
                "cow_id": cow_id,
                "type": "temperature",
                "value": temp,
                "severity": "critical",
                "description": f"Critical high temperature: {temp:.1f}°C"
            })
        elif temp > 39.0:
            alerts.append({
                "cow_id": cow_id,
                "type": "temperature",
                "value": temp,
                "severity": "warning",
                "description": f"Elevated temperature: {temp:.1f}°C"
            })

        # Enhanced heart rate alerts
        if bpm > 110:
            alerts.append({
                "cow_id": cow_id,
                "type": "bpm",
                "value": bpm,
                "severity": "critical",
                "description": f"Critical high heart rate: {bpm} BPM"
            })
        elif bpm > 90:
            alerts.append({
                "cow_id": cow_id,
                "type": "bpm",
                "value": bpm,
                "severity": "warning",
                "description": f"Elevated heart rate: {bpm} BPM"
            })

        # Enhanced oxygen alerts
        if spo2 < 85:
            alerts.append({
                "cow_id": cow_id,
                "type": "spo2",
                "value": spo2,
                "severity": "critical",
                "description": f"Critical low oxygen: {spo2}%"
            })
        elif spo2 < 92:
            alerts.append({
                "cow_id": cow_id,
                "type": "spo2",
                "value": spo2,
                "severity": "warning",
                "description": f"Low oxygen: {spo2}%"
            })

        # Health status alerts
        if health_status == 'CRITICAL':
            alerts.append({
                "cow_id": cow_id,
                "type": "health_status",
                "value": "CRITICAL",
                "severity": "critical",
                "description": "Critical health status detected"
            })
        elif health_status == 'WARNING':
            alerts.append({
                "cow_id": cow_id,
                "type": "health_status",
                "value": "WARNING",
                "severity": "warning",
                "description": "Warning health status"
            })

        # Data quality alerts
        if data_quality < 50:
            alerts.append({
                "cow_id": cow_id,
                "type": "data_quality",
                "value": data_quality,
                "severity": "warning",
                "description": f"Poor data quality: {data_quality:.0f}/100"
            })

    # Cow-wise enhanced analysis
    cow_analysis = {}
    for cow_id, stats in cow_stats.items():
        if not stats['temps']:
            continue

        try:
            # Calculate averages
            avg_temp = np.mean(stats['temps']) if stats['temps'] else 0
            avg_bpm = np.mean(stats['bpms']) if stats['bpms'] else 0
            avg_spo2 = np.mean(stats['spo2s']) if stats['spo2s'] else 0
            avg_health_score = np.mean(stats['health_scores']) if stats['health_scores'] else 100
            avg_data_quality = np.mean(stats['data_qualities']) if stats['data_qualities'] else 100

            # Get latest values
            latest_health_score = stats['health_scores'][-1] if stats['health_scores'] else 100
            latest_health_status = stats['health_statuses'][-1] if stats['health_statuses'] else 'NORMAL'

            # Calculate trends (simple - last vs first of last 5)
            temp_trend = 0
            if len(stats['temps']) >= 2:
                recent_temps = stats['temps'][-5:] if len(stats['temps']) >= 5 else stats['temps']
                if len(recent_temps) >= 2:
                    temp_trend = recent_temps[-1] - recent_temps[0]

            bpm_trend = 0
            if len(stats['bpms']) >= 2:
                recent_bpms = stats['bpms'][-5:] if len(stats['bpms']) >= 5 else stats['bpms']
                if len(recent_bpms) >= 2:
                    bpm_trend = recent_bpms[-1] - recent_bpms[0]

            health_trend = 0
            if len(stats['health_scores']) >= 2:
                recent_scores = stats['health_scores'][-5:] if len(stats['health_scores']) >= 5 else stats['health_scores']
                if len(recent_scores) >= 2:
                    health_trend = recent_scores[-1] - recent_scores[0]

            cow_analysis[cow_id] = {
                'basic_stats': {
                    'avg_temp': round(avg_temp, 2),
                    'avg_bpm': round(avg_bpm, 1),
                    'avg_spo2': round(avg_spo2, 1),
                    'avg_health_score': round(avg_health_score, 1),
                    'avg_data_quality': round(avg_data_quality, 1),
                    'readings_count': len(stats['temps'])
                },
                'current_status': {
                    'health_score': round(latest_health_score, 1),
                    'health_status': latest_health_status,
                    'health_description': calculate_health_status(latest_health_score)[0],
                    'health_class': calculate_health_status(latest_health_score)[1]
                },
                'trends': {
                    'temperature': {
                        'value': round(temp_trend, 2),
                        'icon': get_trend_icon(temp_trend)[0],
                        'class': get_trend_icon(temp_trend)[1],
                        'description': get_trend_icon(temp_trend)[2]
                    },
                    'heart_rate': {
                        'value': round(bpm_trend, 1),
                        'icon': get_trend_icon(bpm_trend/10)[0],  # Scale for trend
                        'class': get_trend_icon(bpm_trend/10)[1],
                        'description': get_trend_icon(bpm_trend/10)[2]
                    },
                    'health_score': {
                        'value': round(health_trend, 1),
                        'icon': get_trend_icon(health_trend/10)[0],  # Scale for trend
                        'class': get_trend_icon(health_trend/10)[1],
                        'description': get_trend_icon(health_trend/10)[2]
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing cow {cow_id}: {e}")
            continue

    # Overall statistics
    try:
        all_temps = [v for s in cow_stats.values() for v in s['temps']]
        all_bpms = [v for s in cow_stats.values() for v in s['bpms']]
        all_spo2s = [v for s in cow_stats.values() for v in s['spo2s']]
        all_health_scores = [v for s in cow_stats.values() for v in s['health_scores']]
        all_data_qualities = [v for s in cow_stats.values() for v in s['data_qualities']]

        return {
            'total_records': len(data),
            'avg_temperature': round(np.mean(all_temps), 2) if all_temps else 0,
            'avg_bpm': round(np.mean(all_bpms), 1) if all_bpms else 0,
            'avg_spo2': round(np.mean(all_spo2s), 1) if all_spo2s else 0,
            'avg_health_score': round(np.mean(all_health_scores), 1) if all_health_scores else 100,
            'avg_data_quality': round(np.mean(all_data_qualities), 1) if all_data_qualities else 100,
            'max_temperature': round(max(all_temps), 2) if all_temps else 0,
            'min_temperature': round(min(all_temps), 2) if all_temps else 0,
            'alerts_count': len(alerts),
            'critical_alerts': sum(1 for a in alerts if a['severity'] == 'critical'),
            'warning_alerts': sum(1 for a in alerts if a['severity'] == 'warning'),
            'alerts_list': alerts[:10],  # Limit to 10 most recent
            'cow_analysis': cow_analysis,
            'active_cows': len(cow_analysis),
            'latest_timestamp': data[0].get('created_at', 'No data') if data else 'No data',
            'system_status': 'operational' if len(data) > 0 else 'no_data'
        }
    except Exception as e:
        logger.error(f"Error calculating overall statistics: {e}")
        return {
            'total_records': len(data),
            'error': str(e),
            'system_status': 'error'
        }

@cache_with_ttl(ttl_seconds=30)
def fetch_enhanced_data(limit=20):
    """Fetch enhanced data from Supabase"""
    try:
        logger.info(f"Fetching data from Supabase: {SUPABASE_URL}")
        url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
        params = {
            'order': 'created_at.desc',
            'limit': limit
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)

        logger.info(f"Supabase response status: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"Supabase error {response.status_code}: {response.text}")
            return []

        data = response.json()
        logger.info(f"Received {len(data) if isinstance(data, list) else 'non-list'} records")

        if not isinstance(data, list):
            logger.error(f"Unexpected data format: {type(data)}. Content: {data[:500] if data else 'Empty'}")
            return []

        # Filter only items with valid sensor data
        enhanced_data = []
        for idx, item in enumerate(data):
            # Check if we have at least one sensor reading
            if any([
                item.get('temperature') is not None,
                item.get('bpm') is not None,
                item.get('spo2') is not None
            ]):
                # Ensure all enhanced fields have defaults
                item.setdefault('health_score', 100)
                item.setdefault('health_status', 'NORMAL')
                item.setdefault('data_quality', 100)
                item.setdefault('signal_quality', 100)
                item.setdefault('temp_trend', 0.0)
                item.setdefault('perfusion_index', 0.0)
                enhanced_data.append(item)
            else:
                logger.debug(f"Skipping item {idx} - no sensor data")

        logger.info(f"Filtered to {len(enhanced_data)} enhanced records")
        return enhanced_data

    except requests.exceptions.Timeout:
        logger.error("Supabase request timed out")
        return []
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Supabase")
        return []
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return []

# --------------------------
# Routes
# --------------------------
@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_enhanced_data():
    """Enhanced API endpoint"""
    try:
        # Get parameters
        limit = int(request.args.get('limit', 20))
        logger.info(f"API request for /api/data with limit={limit}")

        # Fetch data
        data = fetch_enhanced_data(limit=limit)
        logger.info(f"Fetched {len(data)} records from cache/API")

        if not data:
            logger.warning("No data returned from fetch_enhanced_data")
            return jsonify({
                'success': True,
                'data': [],
                'statistics': {
                    'total_records': 0,
                    'system_status': 'no_data',
                    'message': 'No enhanced data available'
                },
                'last_updated': datetime.utcnow().isoformat() + 'Z'
            })

        # Calculate enhanced statistics
        stats = calculate_enhanced_statistics(data)
        logger.info(f"Calculated statistics: {stats.get('total_records', 0)} records")

        return jsonify({
            'success': True,
            'data': data[:limit],
            'statistics': stats,
            'metadata': {
                'limit': limit,
                'total_available': len(data),
                'enhanced_fields': True,
                'fields_included': [
                    'health_score', 'health_status', 'data_quality',
                    'signal_quality', 'temp_trend', 'perfusion_index'
                ]
            },
            'last_updated': datetime.utcnow().isoformat() + 'Z'
        })

    except Exception as e:
        logger.error(f"API error in get_enhanced_data: {e}")
        return jsonify({
            'success': False,
            'error': f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        logger.info("Health check requested")
        # Test Supabase connection
        response = requests.get(f"{SUPABASE_URL}/rest/v1/", headers=headers, timeout=5)
        logger.info(f"Health check Supabase status: {response.status_code}")

        return jsonify({
            'status': 'healthy' if response.status_code == 200 else 'unhealthy',
            'supabase_connected': response.status_code == 200,
            'supabase_status': response.status_code,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 500

# --------------------------
# Error Handlers
# --------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

# --------------------------
# Run App (for local testing only)
# --------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))

    print("=" * 60)
    print("🐄 ENHANCED CATTLE HEALTH DASHBOARD")
    print("=" * 60)
    print(f"Supabase URL: {SUPABASE_URL}")
    print(f"API Endpoint: /api/data")
    print(f"Health Check: /api/health")
    print("=" * 60)

    app.run(host='0.0.0.0', port=port, debug=False)