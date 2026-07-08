"""
Curated knowledge base for RAG retrieval.

Small, hand-written corpus of activity/destination documents. In a real
deployment this would be sourced from a travel content provider, scraped
guides, or user-generated reviews and refreshed regularly -- this is a
deliberately small seed set to demonstrate the retrieval pattern correctly
without needing a live content pipeline. Swap `DOCUMENTS` for a loader
that reads from a real content source when one exists; nothing else in
the RAG pipeline needs to change.

Each document is grounded content the Recommendation Engine's judgment
layer can cite from -- the point of RAG here is the same as the Google
Places integration: give the model real material to reason over instead
of relying on its own (possibly outdated, possibly invented) training
knowledge about what's worth doing somewhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    id: str
    destination: str          # matches trip.destination.confirmed values
    title: str
    text: str
    tags: list[str]           # matches interests / AI-derived score keys loosely


DOCUMENTS: list[Document] = [
    # --- Agra ---
    Document(
        id="agra_taj_sunrise",
        destination="Agra",
        title="Taj Mahal at sunrise",
        text=(
            "Arrive at the east or west gate by 5:45am for sunrise entry. "
            "The marble shifts from soft pink to brilliant white as the "
            "sun rises, and crowds are thinnest in the first hour. The "
            "reflecting pools offer the classic symmetrical photograph."
        ),
        tags=["history", "photography", "culture", "iconic"],
    ),
    Document(
        id="agra_agra_fort",
        destination="Agra",
        title="Agra Fort",
        text=(
            "A red sandstone Mughal fortress on the Yamuna river, UNESCO "
            "listed, with a clear view of the Taj Mahal from its ramparts. "
            "Less crowded than the Taj itself, good for 90 minutes to 2 hours."
        ),
        tags=["history", "culture", "photography"],
    ),
    Document(
        id="agra_mehtab_bagh",
        destination="Agra",
        title="Mehtab Bagh (sunset garden)",
        text=(
            "A moonlight garden directly across the river from the Taj "
            "Mahal, best visited at sunset for a golden-hour view without "
            "the crowds inside the monument itself. Quiet, good for families "
            "with young kids who need open space to run around."
        ),
        tags=["nature", "relaxation", "photography", "family"],
    ),
    Document(
        id="agra_food_street",
        destination="Agra",
        title="Sadar Bazaar street food",
        text=(
            "Agra's petha (a soft translucent sweet) is a local specialty, "
            "along with bedai (spiced fried bread) for breakfast. Sadar "
            "Bazaar has dense street-food stalls -- go hungry, prices are low."
        ),
        tags=["food", "local_experience", "budget"],
    ),
    Document(
        id="agra_fatehpur_sikri",
        destination="Agra",
        title="Fatehpur Sikri day trip",
        text=(
            "An abandoned Mughal capital about 40km from Agra, roughly a "
            "half-day round trip. Extensive walking on uneven stone "
            "surfaces -- not ideal for travelers with mobility limitations."
        ),
        tags=["history", "culture", "adventure"],
    ),
    Document(
        id="agra_hospital_note",
        destination="Agra",
        title="Medical access near central Agra",
        text=(
            "District Hospital Agra and several private clinics are "
            "concentrated near the Idgah/railway station area, useful to "
            "know when traveling with elderly family members."
        ),
        tags=["health", "safety", "senior_citizen"],
    ),

    # --- Kerala ---
    Document(
        id="kerala_backwaters",
        destination="Kerala",
        title="Alleppey backwater houseboat",
        text=(
            "An overnight houseboat cruise through Kerala's backwaters is "
            "the signature experience -- calm water, coconut groves, and "
            "village life along the banks. Low physical exertion, suitable "
            "for elderly travelers."
        ),
        tags=["relaxation", "nature", "senior_citizen", "iconic"],
    ),
    Document(
        id="kerala_munnar_tea",
        destination="Kerala",
        title="Munnar tea plantations",
        text=(
            "Rolling tea estates in the Western Ghats hill country. Cooler "
            "climate than the coast, good for travelers who want scenery "
            "without extreme heat. Moderate walking on plantation paths."
        ),
        tags=["nature", "photography", "relaxation"],
    ),
    Document(
        id="kerala_ayurveda",
        destination="Kerala",
        title="Ayurvedic wellness retreat",
        text=(
            "Kerala is considered the origin point of formal Ayurvedic "
            "practice in India; many resorts offer short wellness programs "
            "(3-7 days) combining massage, diet, and yoga."
        ),
        tags=["relaxation", "wellness", "luxury"],
    ),
    Document(
        id="kerala_seafood",
        destination="Kerala",
        title="Kerala coastal seafood",
        text=(
            "Coconut-based fish curries and fresh prawns are the regional "
            "specialty, especially along the Kochi and Alleppey coastline. "
            "Spice level runs high by default -- worth asking for mild."
        ),
        tags=["food", "local_experience"],
    ),

    # --- Japan / Tokyo ---
    Document(
        id="japan_shibuya",
        destination="Japan",
        title="Shibuya Crossing and Shibuya Sky",
        text=(
            "The world's busiest pedestrian crossing, best photographed "
            "from Shibuya Sky observation deck above it. Very crowded at "
            "peak hours -- go early morning or late evening for a calmer visit."
        ),
        tags=["photography", "iconic", "urban", "crowds"],
    ),
    Document(
        id="japan_meiji_shrine",
        destination="Japan",
        title="Meiji Shrine",
        text=(
            "A forested Shinto shrine in the middle of Tokyo, a quiet "
            "contrast to Shibuya's density. Flat, easy walking paths -- "
            "comfortable for travelers of any mobility level."
        ),
        tags=["culture", "nature", "relaxation", "senior_citizen"],
    ),
    Document(
        id="japan_ramen",
        destination="Japan",
        title="Tokyo ramen culture",
        text=(
            "Ramen shops in Tokyo range from standing-only counters to "
            "sit-down izakaya-style spots. Ticket-machine ordering is "
            "common -- point at pictures if unsure, most machines have no "
            "English."
        ),
        tags=["food", "local_experience", "budget"],
    ),
    Document(
        id="japan_asakusa",
        destination="Japan",
        title="Asakusa and Senso-ji Temple",
        text=(
            "Tokyo's oldest temple, with a shopping street (Nakamise-dori) "
            "leading up to it selling traditional snacks and souvenirs. "
            "Busy but manageable, good half-day activity."
        ),
        tags=["history", "culture", "photography", "shopping"],
    ),

    # --- Goa ---
    Document(
        id="goa_beaches",
        destination="Goa",
        title="North vs South Goa beaches",
        text=(
            "North Goa (Baga, Calangute) is livelier with more nightlife "
            "and water sports; South Goa (Palolem, Agonda) is quieter and "
            "better suited to families or travelers wanting relaxation "
            "over nightlife."
        ),
        tags=["relaxation", "adventure", "family", "beach"],
    ),
    Document(
        id="goa_water_sports",
        destination="Goa",
        title="Water sports in Goa",
        text=(
            "Parasailing, jet-skiing, and banana boat rides are widely "
            "available along North Goa's beaches. Not recommended for "
            "travelers with walking difficulty or heart conditions due to "
            "the physical exertion involved."
        ),
        tags=["adventure", "water_sports"],
    ),
    Document(
        id="goa_portuguese_heritage",
        destination="Goa",
        title="Old Goa Portuguese churches",
        text=(
            "Basilica of Bom Jesus and Se Cathedral reflect Goa's "
            "Portuguese colonial history, both UNESCO World Heritage "
            "listed. A cooler, quieter alternative to the beaches for a "
            "half-day."
        ),
        tags=["history", "culture", "photography"],
    ),

    # --- Delhi ---
    Document(
        id="delhi_red_fort",
        destination="Delhi",
        title="Red Fort",
        text=(
            "A 17th-century Mughal fortress in Old Delhi, UNESCO listed. "
            "Extensive grounds with moderate walking. The evening sound "
            "and light show recounts Mughal history, good for a second visit."
        ),
        tags=["history", "culture", "photography", "iconic"],
    ),
    Document(
        id="delhi_humayuns_tomb",
        destination="Delhi",
        title="Humayun's Tomb",
        text=(
            "Often called a precursor to the Taj Mahal's design, this "
            "garden tomb is quieter and less crowded than Delhi's other "
            "major sites. Flat, easy walking paths throughout the grounds."
        ),
        tags=["history", "culture", "photography", "senior_citizen"],
    ),
    Document(
        id="delhi_chandni_chowk",
        destination="Delhi",
        title="Chandni Chowk food and market walk",
        text=(
            "One of India's oldest markets, dense with street food stalls, "
            "spice shops, and narrow lanes. Paranthe Wali Gali specializes "
            "in stuffed flatbreads. Very crowded and sensory-intense -- "
            "not ideal for travelers who find dense crowds overwhelming."
        ),
        tags=["food", "local_experience", "shopping", "crowds"],
    ),
    Document(
        id="delhi_lotus_temple",
        destination="Delhi",
        title="Lotus Temple",
        text=(
            "A Bahá'í house of worship shaped like a lotus flower, known "
            "for its quiet, contemplative atmosphere regardless of "
            "religious background. Free entry, moderate queue times."
        ),
        tags=["culture", "relaxation", "photography"],
    ),
    Document(
        id="delhi_connaught_place",
        destination="Delhi",
        title="Connaught Place",
        text=(
            "Central Delhi's colonial-era shopping and dining hub, a ring "
            "of white colonnaded buildings. Good base for restaurants "
            "spanning street food to fine dining, and generally easier "
            "walking than the old city's narrower lanes."
        ),
        tags=["shopping", "food", "urban"],
    ),
    Document(
        id="delhi_mobility_note",
        destination="Delhi",
        title="Getting around Delhi",
        text=(
            "The Delhi Metro is extensive, air-conditioned, and generally "
            "the most reliable way to cross the city, with elevator access "
            "at most major stations. Auto-rickshaws and ride-hailing apps "
            "cover the rest."
        ),
        tags=["transportation", "senior_citizen", "budget"],
    ),

    # --- Paris ---
    Document(
        id="paris_louvre",
        destination="Paris",
        title="The Louvre",
        text=(
            "The world's largest art museum -- realistically needs at "
            "least half a day, ideally a full day. Pre-booking a timed "
            "entry slot avoids the worst of the queue."
        ),
        tags=["culture", "history", "iconic"],
    ),
    Document(
        id="paris_montmartre",
        destination="Paris",
        title="Montmartre and Sacré-Cœur",
        text=(
            "A hillside neighborhood with steep, cobbled streets -- "
            "beautiful but genuinely difficult for travelers with walking "
            "difficulty; the funicular railway is a good alternative to "
            "climbing the main staircase."
        ),
        tags=["culture", "photography", "walking_difficulty"],
    ),
    Document(
        id="paris_patisserie",
        destination="Paris",
        title="Parisian patisserie culture",
        text=(
            "Croissants and pastries are best bought fresh in the morning "
            "-- most patisseries sell out of the best items by early "
            "afternoon. Neighborhood bakeries are often better value than "
            "famous tourist-district ones."
        ),
        tags=["food", "local_experience"],
    ),
]


def documents_for_destination(destination: str) -> list[Document]:
    """Filter the corpus to a single destination (case-insensitive)."""
    key = destination.strip().lower()
    return [d for d in DOCUMENTS if d.destination.strip().lower() == key]
