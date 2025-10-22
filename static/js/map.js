// Initialize the map
let map = L.map('map').setView([50.0755, 14.4378], 13); // Center on Prague

// Add OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors'
}).addTo(map);