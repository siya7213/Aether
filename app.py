import os
import json
import math
import heapq
import datetime
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from haversine import haversine, Unit
from sklearn.cluster import DBSCAN
from deep_translator import GoogleTranslator
from gtts import gTTS

# ==========================================
# 0. PAGE CONFIGURATION & DARK THEME
# ==========================================
st.set_page_config(
    page_title="Aether | Urban Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0e17; color: #e2e8f0; }
    .header-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.8));
        backdrop-filter: blur(12px); border: 1px solid #00f3ff;
        border-radius: 10px; padding: 15px 25px; margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 243, 255, 0.15);
    }
    .metric-box {
        background: rgba(30, 41, 59, 0.7); border-left: 4px solid #00f3ff;
        padding: 12px 18px; border-radius: 6px; margin-bottom: 10px;
    }
    .badge-verified {
        background-color: #10b981; color: #ffffff; padding: 2px 8px;
        border-radius: 12px; font-size: 0.75rem; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATABASE MODELS & DYNAMIC MULTI-CITY REGISTRY
# ==========================================
DB_PATH = "aether.db"
Base = declarative_base()

class City(Base):
    __tablename__ = "cities"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    state = Column(String(100), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    pois = relationship("POI", back_populates="city", cascade="all, delete-orphan")
    landmarks = relationship("Landmark", back_populates="city", cascade="all, delete-orphan")
    neighborhoods = relationship("Neighborhood", back_populates="city", cascade="all, delete-orphan")
    incidents = relationship("IncidentReport", back_populates="city", cascade="all, delete-orphan")

class POI(Base):
    __tablename__ = "pois"
    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey("cities.id"), nullable=False)
    name = Column(String(150), nullable=False)
    category = Column(String(50), nullable=False)
    price_level = Column(Integer, nullable=False)
    open_hour = Column(Integer, nullable=False)
    close_hour = Column(Integer, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    address = Column(String(255), nullable=True)
    city = relationship("City", back_populates="pois")
    reviews = relationship("Review", back_populates="poi", cascade="all, delete-orphan")

class Landmark(Base):
    __tablename__ = "landmarks"
    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey("cities.id"), nullable=False)
    name = Column(String(150), nullable=False)
    category = Column(String(100), nullable=False)
    era = Column(String(100), nullable=False)
    architectural_style = Column(String(100), nullable=False)
    historical_significance = Column(Text, nullable=False)
    traditions = Column(Text, nullable=False)
    best_time = Column(String(100), nullable=False)
    local_tip = Column(Text, nullable=False)
    audio_script = Column(Text, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    city = relationship("City", back_populates="landmarks")

class Neighborhood(Base):
    __tablename__ = "neighborhoods"
    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey("cities.id"), nullable=False)
    name = Column(String(100), nullable=False)
    centroid_lat = Column(Float, nullable=False)
    centroid_lon = Column(Float, nullable=False)
    polygon_geojson = Column(Text, nullable=False)
    safety_score = Column(Float, default=75.0)
    air_quality_index = Column(Float, default=60.0)
    affordability_score = Column(Float, default=70.0)
    transit_score = Column(Float, default=80.0)
    resident_rating = Column(Float, default=4.2)
    city = relationship("City", back_populates="neighborhoods")

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True)
    poi_id = Column(Integer, ForeignKey("pois.id"), nullable=False)
    author_name = Column(String(100), nullable=False)
    is_verified_local = Column(Boolean, default=False)
    rating = Column(Float, nullable=False)
    comment = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    poi = relationship("POI", back_populates="reviews")

class IncidentReport(Base):
    __tablename__ = "incident_reports"
    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey("cities.id"), nullable=False)
    category = Column(String(100), nullable=False)
    severity = Column(Integer, default=1)
    description = Column(Text, nullable=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    city = relationship("City", back_populates="incidents")

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

INDIAN_CITIES_REGISTRY = {
    "Mumbai, Maharashtra": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "Delhi NCR (New Delhi)": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    "Bengaluru, Karnataka": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Hyderabad, Telangana": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Chennai, Tamil Nadu": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "Kolkata, West Bengal": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "Ahmedabad, Gujarat": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
    "Pune, Maharashtra": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
    "Solapur, Maharashtra": {"lat": 17.6599, "lon": 75.9064, "state": "Maharashtra"},
    "Surat, Gujarat": {"lat": 21.1702, "lon": 72.8311, "state": "Gujarat"},
    "Jaipur, Rajasthan": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Lucknow, Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    "Chandigarh, UT": {"lat": 30.7333, "lon": 76.7794, "state": "Chandigarh"},
    "Kochi, Kerala": {"lat": 9.9312, "lon": 76.2673, "state": "Kerala"},
    "Indore, Madhya Pradesh": {"lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh"},
    "Bhopal, Madhya Pradesh": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh"},
    "Visakhapatnam, Andhra Pradesh": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh"},
    "Nagpur, Maharashtra": {"lat": 21.1458, "lon": 79.0882, "state": "Maharashtra"},
    "Coimbatore, Tamil Nadu": {"lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu"},
    "Patna, Bihar": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    "Bhubaneswar, Odisha": {"lat": 20.2961, "lon": 85.8245, "state": "Odisha"},
    "Guwahati, Assam": {"lat": 26.1445, "lon": 91.7362, "state": "Assam"},
    "Dehradun, Uttarakhand": {"lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand"},
    "Varanasi, Uttar Pradesh": {"lat": 25.3176, "lon": 82.9739, "state": "Uttar Pradesh"},
    "Amritsar, Punjab": {"lat": 31.6340, "lon": 74.8723, "state": "Punjab"},
    "Agra, Uttar Pradesh": {"lat": 27.1767, "lon": 78.0081, "state": "Uttar Pradesh"},
    "Nashik, Maharashtra": {"lat": 19.9975, "lon": 73.7898, "state": "Maharashtra"},
    "Thiruvananthapuram, Kerala": {"lat": 8.5241, "lon": 76.9366, "state": "Kerala"},
    "Madurai, Tamil Nadu": {"lat": 9.9252, "lon": 78.1198, "state": "Tamil Nadu"},
    "Vijayawada, Andhra Pradesh": {"lat": 16.5062, "lon": 80.6480, "state": "Andhra Pradesh"},
    "Ranchi, Jharkhand": {"lat": 23.3441, "lon": 85.3096, "state": "Jharkhand"},
    "Raipur, Chhattisgarh": {"lat": 21.2514, "lon": 81.6296, "state": "Chhattisgarh"},
    "Jodhpur, Rajasthan": {"lat": 26.2389, "lon": 73.0243, "state": "Rajasthan"},
    "Mangaluru, Karnataka": {"lat": 12.9141, "lon": 74.8560, "state": "Karnataka"},
    "Puducherry, UT": {"lat": 11.9416, "lon": 79.8083, "state": "Puducherry"},
    "Srinagar, Jammu & Kashmir": {"lat": 34.0837, "lon": 74.7973, "state": "Jammu & Kashmir"},
    "Panaji, Goa": {"lat": 15.4909, "lon": 73.8278, "state": "Goa"}
}

def generate_dynamic_city_data(city_display_name: str):
    info = INDIAN_CITIES_REGISTRY.get(city_display_name, {"lat": 18.5204, "lon": 73.8567, "state": "India"})
    base_lat, base_lon = info["lat"], info["lon"]
    clean_name = city_display_name.split(",")[0].strip()

    pois = [
        {"name": f"Central Cafe - {clean_name}", "category": "Cafe", "price_level": 2, "open_hour": 8, "close_hour": 23, "lat": base_lat + 0.003, "lon": base_lon + 0.002, "address": f"Main Road, {clean_name}"},
        {"name": f"Heritage Mess & Dining", "category": "Restaurant", "price_level": 1, "open_hour": 11, "close_hour": 22, "lat": base_lat - 0.002, "lon": base_lon - 0.003, "address": f"Station Square, {clean_name}"},
        {"name": f"{clean_name} Street Food Hub", "category": "Street Stall", "price_level": 1, "open_hour": 16, "close_hour": 23, "lat": base_lat + 0.001, "lon": base_lon - 0.002, "address": f"Gandhi Chowk, {clean_name}"},
        {"name": f"Grand Palace Hotel", "category": "Restaurant", "price_level": 3, "open_hour": 7, "close_hour": 23, "lat": base_lat - 0.004, "lon": base_lon + 0.005, "address": f"Civil Lines, {clean_name}"},
        {"name": f"City Backpackers Hostel", "category": "Hostel", "price_level": 1, "open_hour": 0, "close_hour": 24, "lat": base_lat + 0.005, "lon": base_lon - 0.001, "address": f"University Road, {clean_name}"}
    ]

    landmarks = [
        {"name": f"Historic Fort of {clean_name}", "category": "Architectural & Monumental Heritage", "era": "16th-18th Century", "architectural_style": "Vernacular Fortification", "historical_significance": f"Historical citadel protecting trade corridors across {info['state']}.", "traditions": "Hosts annual civic pageants and heritage drives.", "best_time": "5:00 PM - 7:00 PM", "local_tip": "Visit at sunset for skyline views.", "audio_script": f"Welcome to the historic citadel of {clean_name}.", "lat": base_lat + 0.002, "lon": base_lon + 0.001},
        {"name": f"Central Temple Shrine", "category": "Sacred & Spiritual Landmarks", "era": "Ancient Era", "architectural_style": "Traditional Indian Temple Style", "historical_significance": f"Sacred spiritual center revered for centuries across {clean_name}.", "traditions": "Site of daily morning aarti rituals.", "best_time": "6:00 AM or 7:00 PM", "local_tip": "Remove footwear at outer gates.", "audio_script": f"You are visiting the central shrine of {clean_name}.", "lat": base_lat - 0.001, "lon": base_lon + 0.003},
        {"name": f"Old Town Handloom Bazaar", "category": "Living Heritage & Artisan Precincts", "era": "19th Century", "architectural_style": "Heritage Market Precinct", "historical_significance": f"Traditional bazaar preserving generational artisan crafts of {info['state']}.", "traditions": "Bustling market for regional textiles.", "best_time": "11:00 AM - 6:00 PM", "local_tip": "Bargain respectfully with local shopkeepers.", "audio_script": f"Explore the living handloom bazaar of {clean_name}.", "lat": base_lat - 0.003, "lon": base_lon - 0.002},
        {"name": f"Iconic Heritage Sweet House", "category": "Culinary Heritage & Iconic Food Hubs", "era": "20th Century", "architectural_style": "Vernacular Culinary Mess", "historical_significance": f"Legendary eatery preserving authentic recipes of {clean_name}.", "traditions": "Famous for traditional festival sweet preparations.", "best_time": "12:00 PM - 3:00 PM", "local_tip": "Try the signature local savory dish.", "audio_script": f"Welcome to {clean_name}'s iconic culinary house.", "lat": base_lat + 0.004, "lon": base_lon - 0.004},
        {"name": f"{clean_name} Government Museum", "category": "Museums, Galleries & Knowledge Hubs", "era": "1950s", "architectural_style": "Post-Independence Civic Architecture", "historical_significance": f"Houses artifacts detailing regional freedom movements in {clean_name}.", "traditions": "Hosts educational exhibitions for youth.", "best_time": "10:00 AM - 4:00 PM", "local_tip": "Student ID cards offer entry discounts.", "audio_script": f"Examine centuries of history inside the {clean_name} Museum.", "lat": base_lat + 0.001, "lon": base_lon + 0.004},
        {"name": f"Central Park Reservoir", "category": "Natural & Ecological Heritage", "era": "19th Century", "architectural_style": "Ecological Reserve & Catchment", "historical_significance": f"Historic water catchment and green lung for {clean_name}.", "traditions": "Popular venue for morning walking clubs.", "best_time": "6:00 AM - 8:00 AM", "local_tip": "Visit the eastern promenade for bird watching.", "audio_script": f"Enjoy the natural park reserve of {clean_name}.", "lat": base_lat - 0.005, "lon": base_lon + 0.001}
    ]

    neighborhoods = [
        {"name": f"{clean_name} Central", "centroid_lat": base_lat + 0.001, "centroid_lon": base_lon + 0.001, "polygon_geojson": json.dumps({"type": "Polygon", "coordinates": [[[base_lon, base_lat], [base_lon+0.01, base_lat], [base_lon+0.01, base_lat+0.01], [base_lon, base_lat+0.01], [base_lon, base_lat]]]}), "safety_score": 85.0, "air_quality_index": 60.0, "affordability_score": 70.0, "transit_score": 90.0, "resident_rating": 4.5},
        {"name": f"Civic Lines", "centroid_lat": base_lat - 0.002, "centroid_lon": base_lon + 0.003, "polygon_geojson": json.dumps({"type": "Polygon", "coordinates": [[[base_lon, base_lat], [base_lon-0.01, base_lat], [base_lon-0.01, base_lat-0.01], [base_lon, base_lat-0.01], [base_lon, base_lat]]]}), "safety_score": 88.0, "air_quality_index": 65.0, "affordability_score": 60.0, "transit_score": 85.0, "resident_rating": 4.6},
        {"name": f"Market Heights", "centroid_lat": base_lat + 0.004, "centroid_lon": base_lon - 0.002, "polygon_geojson": json.dumps({"type": "Polygon", "coordinates": [[[base_lon, base_lat], [base_lon+0.01, base_lat], [base_lon+0.01, base_lat-0.01], [base_lon, base_lat-0.01], [base_lon, base_lat]]]}), "safety_score": 78.0, "air_quality_index": 55.0, "affordability_score": 85.0, "transit_score": 88.0, "resident_rating": 4.2},
        {"name": f"Industrial Zone", "centroid_lat": base_lat - 0.005, "centroid_lon": base_lon - 0.005, "polygon_geojson": json.dumps({"type": "Polygon", "coordinates": [[[base_lon, base_lat], [base_lon-0.01, base_lat], [base_lon-0.01, base_lat+0.01], [base_lon, base_lat+0.01], [base_lon, base_lat]]]}), "safety_score": 75.0, "air_quality_index": 48.0, "affordability_score": 80.0, "transit_score": 75.0, "resident_rating": 4.0}
    ]

    return {"city": {"name": clean_name, "state": info["state"], "lat": base_lat, "lon": base_lon}, "pois": pois, "landmarks": landmarks, "neighborhoods": neighborhoods}

# ==========================================
# 2. LOCALIZATION & TRANSLATION ENGINE
# ==========================================
SUPPORTED_LANGUAGES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "ta": "Tamil", "te": "Telugu",
    "bn": "Bengali", "kn": "Kannada", "gu": "Gujarati", "ml": "Malayalam", "pa": "Punjabi"
}

STATIC_STRINGS = {
    "en": {"app_title": "Aether: Urban Intelligence Platform", "city_selector": "Select City", "language_selector": "Select Language", "tab_explore": "Explore & Hospitality", "tab_heritage": "History & Heritage", "tab_safety": "Safe Navigation", "tab_compare": "City Analytics", "tab_triage": "Citizen Triage", "tab_voice": "Voice Assistant", "tourist_pulse": "Tourist Pulse", "residential_pulse": "Residential Pulse (Verified Locals)", "verified_local_badge": "Verified Resident Local", "simulate_ping": "Simulate 30 Days Pings", "budget_estimator": "Smart Budget Estimator", "safest_route": "Safest Route", "shortest_route": "Shortest Route", "livability_index": "Livability Index Score"},
    "hi": {"app_title": "एथर: शहरी इंटेलिजेंस प्लेटफॉर्म", "city_selector": "शहर चुनें", "language_selector": "भाषा चुनें", "tab_explore": "खोजें और आतिथ्य", "tab_heritage": "इतिहास और विरासत", "tab_safety": "सुरक्षित नेविगेशन", "tab_compare": "शहर विश्लेषण", "tab_triage": "नागरिक रिपोर्ट", "tab_voice": "वॉयस असिस्टेंट", "tourist_pulse": "पर्यटक समीक्षाएं", "residential_pulse": "स्थानीय नागरिक समीक्षाएं (सत्यापित)", "verified_local_badge": "सत्यापित स्थानीय निवासी", "simulate_ping": "30 दिनों का स्थान सिम्यूलेट करें", "budget_estimator": "स्मार्ट बजट अनुमानक", "safest_route": "सबसे सुरक्षित रास्ता", "shortest_route": "सबसे छोटा रास्ता", "livability_index": "रहने योग्य सूचकांक स्कोर"},
    "mr": {"app_title": "एथर: नागरी बुद्धिमत्ता मंच", "city_selector": "शहर निवडा", "language_selector": "भाषा निवडा", "tab_explore": "शोधा आणि आदरातिथ्य", "tab_heritage": "इतिहास आणि वारसा", "tab_safety": "सुरक्षित मार्ग", "tab_compare": "शहर विश्लेषण", "tab_triage": "नागरिक तक्रार", "tab_voice": "व्हॉइस असिस्टंट", "tourist_pulse": "पर्यटक पुनरावलोकने", "residential_pulse": "स्थानिक नागरिक मते (प्रमाणित)", "verified_local_badge": "प्रमाणित स्थानिक रहिवासी", "simulate_ping": "३० दिवसांचे लोकेशन सिम्युलेट करा", "budget_estimator": "स्मार्ट बजेट अंदाजक", "safest_route": "सर्वात सुरक्षित मार्ग", "shortest_route": "सर्वात छोटा मार्ग", "livability_index": "राहणीमान निर्देशांक गुण"}
}

def t(key):
    lang = st.session_state.get("selected_lang", "en")
    return STATIC_STRINGS.get(lang, STATIC_STRINGS["en"]).get(key, STATIC_STRINGS["en"].get(key, key))

def tr(text):
    lang = st.session_state.get("selected_lang", "en")
    if not text or lang == "en":
        return text
    try:
        translator = GoogleTranslator(source="auto", target=lang)
        return translator.translate(text)
    except Exception:
        return text

# ==========================================
# 3. SAFETY SCORING & A* ROUTING ENGINE
# ==========================================
def calculate_edge_safety(lat1, lon1, lat2, lon2, incidents, festival_active=False):
    mid_lat, mid_lon = (lat1 + lat2) / 2.0, (lon1 + lon2) / 2.0
    police_dist_m = haversine((mid_lat, mid_lon), (mid_lat + 0.005, mid_lon + 0.005), unit=Unit.METERS)
    police_prox = math.exp(-police_dist_m / 800.0)
    S_base = 0.25 * 0.8 + 0.25 * 0.7 + 0.20 * police_prox + 0.10 * 0.6 + 0.20 * 0.9
    
    inc_count = sum(1 for inc in incidents if haversine((mid_lat, mid_lon), (inc.lat, inc.lon), unit=Unit.METERS) <= 300)
    b = inc_count / (inc_count + 5.0)
    S_incident = max(0.0, 1.0 - (inc_count * 0.15))
    S_final = (1.0 - b) * S_base + b * S_incident
    
    if festival_active and inc_count > 0:
        S_final *= 0.2
    return min(1.0, max(0.0, S_final))

def build_grid_graph(city_lat, city_lon, incidents, festival_active=False):
    grid_size = 5
    lat_step, lon_step = 0.004, 0.004
    nodes, edges = {}, {}
    
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = f"{i}_{j}"
            nodes[node_id] = (city_lat + (i - 2) * lat_step, city_lon + (j - 2) * lon_step)
            edges[node_id] = []

    for i in range(grid_size):
        for j in range(grid_size):
            u = f"{i}_{j}"
            neighbors = []
            if i + 1 < grid_size: neighbors.append(f"{i+1}_{j}")
            if j + 1 < grid_size: neighbors.append(f"{i}_{j+1}")
            for v in neighbors:
                d = haversine(nodes[u], nodes[v], unit=Unit.METERS)
                S_e = calculate_edge_safety(nodes[u][0], nodes[u][1], nodes[v][0], nodes[v][1], incidents, festival_active)
                edges[u].append((v, d, S_e))
                edges[v].append((u, d, S_e))
                
    return nodes, edges

def a_star_search(nodes, edges, start_node, goal_node, alpha=0.5):
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    came_from = {}
    g_score = {node: float('inf') for node in nodes}
    g_score[start_node] = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_node:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor, d, S_e in edges[current]:
            weight = d * (1.0 + alpha * (1.0 - S_e) * 3.0)
            tentative_g = g_score[current] + weight
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + haversine(nodes[neighbor], nodes[goal_node], unit=Unit.METERS)
                heapq.heappush(open_set, (f_score, neighbor))
                
    return list(nodes.keys())[:3]

# ==========================================
# 4. INITIALIZE SESSION STATE
# ==========================================
if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = "Mumbai, Maharashtra"
if "selected_lang" not in st.session_state:
    st.session_state["selected_lang"] = "en"
if "verified_local" not in st.session_state:
    st.session_state["verified_local"] = False

# ==========================================
# 5. HEADER & GLOBAL CONTROLS
# ==========================================
st.markdown('<div class="header-card">', unsafe_allow_html=True)
col_h1, col_h2, col_h3 = st.columns([2, 1, 1])

with col_h1:
    st.title("⚡ Aether")
    st.caption("Smart City Exploration & Intelligence Platform for India")

with col_h2:
    city_options = list(INDIAN_CITIES_REGISTRY.keys())
    city_choice = st.selectbox("📍 " + t("city_selector"), options=city_options, index=city_options.index(st.session_state["selected_city"]))
    
    if city_choice != st.session_state["selected_city"]:
        st.session_state["selected_city"] = city_choice
        st.rerun()

with col_h3:
    lang_codes = list(SUPPORTED_LANGUAGES.keys())
    lang_labels = [f"{v} ({k})" for k, v in SUPPORTED_LANGUAGES.items()]
    curr_idx = lang_codes.index(st.session_state["selected_lang"]) if st.session_state["selected_lang"] in lang_codes else 0
    selected_lang_idx = st.selectbox("🌐 " + t("language_selector"), range(len(lang_codes)), format_func=lambda i: lang_labels[i], index=curr_idx)
    
    if lang_codes[selected_lang_idx] != st.session_state["selected_lang"]:
        st.session_state["selected_lang"] = lang_codes[selected_lang_idx]
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Query Database & Seed Dynamic Data if Missing
db_session = SessionLocal()
clean_city_name = st.session_state["selected_city"].split(",")[0].strip()
active_city = db_session.query(City).filter_by(name=clean_city_name).first()

if not active_city:
    dynamic_data = generate_dynamic_city_data(st.session_state["selected_city"])
    active_city = City(name=dynamic_data["city"]["name"], state=dynamic_data["city"]["state"], lat=dynamic_data["city"]["lat"], lon=dynamic_data["city"]["lon"])
    db_session.add(active_city)
    db_session.flush()

    for p in dynamic_data["pois"]:
        poi = POI(city_id=active_city.id, name=p["name"], category=p["category"], price_level=p["price_level"], open_hour=p["open_hour"], close_hour=p["close_hour"], lat=p["lat"], lon=p["lon"], address=p["address"])
        db_session.add(poi)
        db_session.flush()
        db_session.add(Review(poi_id=poi.id, author_name="Verified Resident", is_verified_local=True, rating=4.7, comment=f"Authentic local venue in {clean_city_name}."))

    for l in dynamic_data["landmarks"]:
        lm = Landmark(city_id=active_city.id, name=l["name"], category=l["category"], era=l["era"], architectural_style=l["architectural_style"], historical_significance=l["historical_significance"], traditions=l["traditions"], best_time=l["best_time"], local_tip=l["local_tip"], audio_script=l["audio_script"], lat=l["lat"], lon=l["lon"])
        db_session.add(lm)

    for n in dynamic_data["neighborhoods"]:
        nb = Neighborhood(city_id=active_city.id, name=n["name"], centroid_lat=n["centroid_lat"], centroid_lon=n["centroid_lon"], polygon_geojson=n["polygon_geojson"], safety_score=n["safety_score"], air_quality_index=n["air_quality_index"], affordability_score=n["affordability_score"], transit_score=n["transit_score"], resident_rating=n["resident_rating"])
        db_session.add(nb)

    db_session.add(IncidentReport(city_id=active_city.id, category="Dark Alley", severity=3, description="Low lighting.", lat=active_city.lat+0.002, lon=active_city.lon+0.002))
    db_session.commit()

# ==========================================
# 6. KPI METRICS DASHBOARD
# ==========================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
avg_safety = sum(n.safety_score for n in active_city.neighborhoods) / len(active_city.neighborhoods) if active_city.neighborhoods else 75.0

with kpi1:
    st.markdown(f'<div class="metric-box"><small>{tr("Active City")}</small><h3>{active_city.name}, {active_city.state}</h3></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown(f'<div class="metric-box"><small>{tr("Safety Index")}</small><h3 style="color:#00f3ff;">{avg_safety:.1f} / 100</h3></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown(f'<div class="metric-box"><small>{tr("Heritage Landmarks")}</small><h3 style="color:#ffb700;">{len(active_city.landmarks)} {tr("Spots")}</h3></div>', unsafe_allow_html=True)
with kpi4:
    st.markdown(f'<div class="metric-box"><small>{tr("Civic Hazards")}</small><h3 style="color:#ef4444;">{len(active_city.incidents)} {tr("Reports")}</h3></div>', unsafe_allow_html=True)

# ==========================================
# 7. MODULE TABS
# ==========================================
tabs = st.tabs([
    "🗺️ " + t("tab_explore"),
    "🏛️ " + t("tab_heritage"),
    "🛡️ " + t("tab_safety"),
    "📊 " + t("tab_compare"),
    "📢 " + t("tab_triage"),
    "🎙️ " + t("tab_voice")
])

# MODULE 1: EXPLORE
with tabs[0]:
    st.subheader(tr("Smart Exploration & Verified Hospitality"))
    col_exp1, col_exp2 = st.columns([1, 2])
    with col_exp1:
        st.markdown(f"#### {tr('Smart Budget Estimator')}")
        user_budget = st.slider(tr("Budget Limit (₹)"), 50, 1000, 200, 50)
        user_time = st.slider(tr("Time of Day (Hour)"), 6, 23, 14, 1)
        matching_pois = [p for p in active_city.pois if {1:100, 2:300, 3:800, 4:1500}.get(p.price_level, 100) <= user_budget and p.open_hour <= user_time <= p.close_hour]
        st.success(f"{len(matching_pois)} {tr('Venues Match Your Budget')}")
        
        st.divider()
        if st.session_state["verified_local"]:
            st.markdown(f'<span class="badge-verified">{tr("Verified Resident Local")}</span>', unsafe_allow_html=True)
        else:
            if st.button(t("simulate_ping")):
                st.session_state["verified_local"] = True
                st.balloons()
                st.rerun()

    with col_exp2:
        df_pois = pd.DataFrame([{"name": p.name, "lat": p.lat, "lon": p.lon, "category": p.category} for p in active_city.pois])
        layer_pois = pdk.Layer("ScatterplotLayer", df_pois, get_position=["lon", "lat"], get_color="[0, 243, 255, 200]", get_radius=120, pickable=True)
        st.pydeck_chart(pdk.Deck(layers=[layer_pois], initial_view_state=pdk.ViewState(latitude=active_city.lat, longitude=active_city.lon, zoom=13), tooltip={"text": "{name}\n{category}"}))
        
        if active_city.pois:
            sel_poi = st.selectbox(tr("Select Venue for Dual Reviews"), active_city.pois, format_func=lambda x: x.name)
            tab_r1, tab_r2 = st.tabs([t("tourist_pulse"), t("residential_pulse")])
            with tab_r1:
                for r in [rev for rev in sel_poi.reviews if not rev.is_verified_local]:
                    st.write(f"⭐ **{r.rating}/5.0** - *{r.author_name}*\n{tr(r.comment)}")
            with tab_r2:
                for r in [rev for rev in sel_poi.reviews if rev.is_verified_local]:
                    st.markdown(f'<span class="badge-verified">{tr("Verified Local")}</span> ⭐ **{r.rating}/5.0** - *{r.author_name}*\n{tr(r.comment)}', unsafe_allow_html=True)

# MODULE 2: HERITAGE DIRECTORY
with tabs[1]:
    st.subheader(tr("6-Category Heritage Directory & Audio Guides"))
    categories = list(set(l.category for l in active_city.landmarks))
    if categories:
        sel_cat = st.radio(tr("Heritage Categories"), categories)
        for lm in [l for l in active_city.landmarks if l.category == sel_cat]:
            with st.expander(f"🏛️ {lm.name} ({lm.era})", expanded=True):
                st.write(f"**{tr('Historical Significance')}:** {tr(lm.historical_significance)}")
                st.write(f"**{tr('Traditions & Customs')}:** {tr(lm.traditions)}")
                st.info(f"💡 **{tr('Local Tip')}:** {tr(lm.local_tip)}")
                if st.button(f"🔊 {tr('Play AI Audio Guide')} ({lm.name})"):
                    tts = gTTS(text=tr(lm.audio_script), lang=st.session_state["selected_lang"] if st.session_state["selected_lang"] in ["en","hi","mr","ta","te","bn","kn","gu","ml","pa"] else "en")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                        tts.save(fp.name)
                        st.audio(fp.name, format="audio/mp3")

# MODULE 3: SAFE PATHWAYS & SMS
with tabs[2]:
    st.subheader(tr("Safety-Weighted A* Routing & Emergency Protocol"))
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        s_weight = st.slider(tr("Safety Preference"), 0.0, 1.0, 0.7, 0.1)
        nodes, edges = build_grid_graph(active_city.lat, active_city.lon, active_city.incidents)
        safest_path = a_star_search(nodes, edges, "0_0", "4_4", alpha=s_weight)
        st.success(tr("Safest Path Computed"))
        
        st.divider()
        if st.button(tr("Generate Emergency QR & SMS Payload")):
            st.code(f"ALERT: Off-route at {active_city.lat},{active_city.lon} in {active_city.name}. Assistance required.", language="text")

    with col_s2:
        path_coords = [nodes[nid] for nid in safest_path]
        layer_path = pdk.Layer("PathLayer", [{"path": [[lon, lat] for lat, lon in path_coords]}], get_color="[0, 243, 255, 255]", width_min_pixels=5)
        st.pydeck_chart(pdk.Deck(layers=[layer_path], initial_view_state=pdk.ViewState(latitude=active_city.lat, longitude=active_city.lon, zoom=13)))

# MODULE 4: ANALYTICS MATRIX
with tabs[3]:
    st.subheader(tr("Neighborhood Livability Comparison Matrix"))
    if active_city.neighborhoods:
        records = []
        for n in active_city.neighborhoods:
            L_i = 100 * (0.30*(n.safety_score/100) + 0.20*(n.air_quality_index/100) + 0.20*(n.affordability_score/100) + 0.20*(n.transit_score/100) + 0.10*(n.resident_rating/5.0))
            records.append({tr("Neighborhood"): n.name, tr("Livability Score"): round(L_i, 1), tr("Safety Index"): n.safety_score, tr("Air Quality"): n.air_quality_index, tr("Affordability"): n.affordability_score})
        df_m = pd.DataFrame(records)
        st.dataframe(df_m, use_container_width=True)

# MODULE 5: CITIZEN TRIAGE
with tabs[4]:
    st.subheader(tr("Citizen Hazard Reporting & DBSCAN Clustering"))
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        with st.form("hazard_form"):
            cat = st.selectbox(tr("Category"), ["Pothole", "Dark Alley", "Waterlogging", "Accident Zone"])
            sev = st.slider(tr("Severity"), 1, 5, 3)
            desc = st.text_area(tr("Description"))
            if st.form_submit_button(tr("Report Hazard")):
                db_session.add(IncidentReport(city_id=active_city.id, category=cat, severity=sev, description=desc, lat=active_city.lat+0.001, lon=active_city.lon+0.001))
                db_session.commit()
                st.success(tr("Hazard reported!"))
                st.rerun()

    with col_t2:
        if active_city.incidents:
            df_inc = pd.DataFrame([{"lat": inc.lat, "lon": inc.lon, "category": inc.category} for inc in active_city.incidents])
            layer_inc = pdk.Layer("ScatterplotLayer", df_inc, get_position=["lon", "lat"], get_color="[239, 68, 68, 200]", get_radius=200)
            st.pydeck_chart(pdk.Deck(layers=[layer_inc], initial_view_state=pdk.ViewState(latitude=active_city.lat, longitude=active_city.lon, zoom=13)))

# MODULE 6: VOICE ASSISTANT
with tabs[5]:
    st.subheader(tr("Voice-First Senior Accessibility Mode"))
    query = st.text_input(tr("Voice Command Input"), placeholder="उदा. मला सर्वात सुरक्षित रस्ता दाखवा")
    if query:
        res = tr(f"Calculating safest route in {active_city.name}.")
        st.write(res)
        tts = gTTS(text=res, lang=st.session_state["selected_lang"] if st.session_state["selected_lang"] in ["en","hi","mr","ta","te","bn","kn","gu","ml","pa"] else "en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")

db_session.close()
