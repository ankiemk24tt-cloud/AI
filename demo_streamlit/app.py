import json
import os
import unicodedata
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st
import pandas as pd
from surprise import dump

HERE = os.path.dirname(os.path.abspath(__file__))
WIKIDATA_QUERY_URL = "https://query.wikidata.org/sparql"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
MOVIE_GENRES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
GENRE_ALIASES = {
    "action": "Action",
    "hanh dong": "Action",
    "adventure": "Adventure",
    "phieu luu": "Adventure",
    "animation": "Animation",
    "hoat hinh": "Animation",
    "children": "Children's",
    "tre em": "Children's",
    "comedy": "Comedy",
    "hai": "Comedy",
    "crime": "Crime",
    "toi pham": "Crime",
    "documentary": "Documentary",
    "tai lieu": "Documentary",
    "drama": "Drama",
    "tam ly": "Drama",
    "chinh kich": "Drama",
    "fantasy": "Fantasy",
    "gia tuong": "Fantasy",
    "film noir": "Film-Noir",
    "horror": "Horror",
    "kinh di": "Horror",
    "musical": "Musical",
    "am nhac": "Musical",
    "mystery": "Mystery",
    "bi an": "Mystery",
    "romance": "Romance",
    "lang man": "Romance",
    "sci fi": "Sci-Fi",
    "khoa hoc vien tuong": "Sci-Fi",
    "thriller": "Thriller",
    "giat gan": "Thriller",
    "war": "War",
    "chien tranh": "War",
    "western": "Western",
    "mien tay": "Western",
}
GENRE_LABELS = {
    "Action": "Hành động",
    "Adventure": "Phiêu lưu",
    "Animation": "Hoạt hình",
    "Children's": "Trẻ em",
    "Comedy": "Hài",
    "Crime": "Tội phạm",
    "Documentary": "Tài liệu",
    "Drama": "Tâm lý",
    "Fantasy": "Giả tưởng",
    "Film-Noir": "Phim noir",
    "Horror": "Kinh dị",
    "Musical": "Âm nhạc",
    "Mystery": "Bí ẩn",
    "Romance": "Lãng mạn",
    "Sci-Fi": "Khoa học viễn tưởng",
    "Thriller": "Giật gân",
    "War": "Chiến tranh",
    "Western": "Miền Tây",
}
st.set_page_config(page_title="Gợi ý phim", page_icon="🎬")


def normalize_text(value):
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(
        char
        for char in normalized
        if not unicodedata.combining(char)
    ).replace("đ", "d")


@st.cache_resource
def load_all():
    model_path = os.path.join(HERE, 'best_svd_model.pkl')
    if not os.path.isfile(model_path):
        model_path = os.path.join(HERE, 'svd_model.pkl')
    _, algo = dump.load(model_path)
    names = {}
    movie_genres = {}
    with open(os.path.join(HERE, 'u.item'), encoding='latin-1') as f:
        for line in f:
            p = line.split('|')
            movie_id = int(p[0])
            names[movie_id] = p[1]
            movie_genres[movie_id] = [
                genre
                for genre, flag in zip(MOVIE_GENRES, p[5:24])
                if flag == "1" and genre != "unknown"
            ]
    return algo, names, movie_genres


@st.cache_data(ttl=3600)
def wikidata_query(query):
    url = f"{WIKIDATA_QUERY_URL}?{urlencode({'query': query, 'format': 'json'})}"
    request = Request(
        url,
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": "MovieRecommenderDemo/1.0 (Streamlit app)",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(
            f"Wikidata trả về lỗi HTTP {error.code}. Vui lòng thử lại sau."
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Không thể kết nối Wikidata. Hãy kiểm tra kết nối mạng: {error.reason}"
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Wikidata trả về dữ liệu không hợp lệ.") from error


@st.cache_data(ttl=3600)
def search_wikidata_people(actor_query):
    params = urlencode(
        {
            "action": "wbsearchentities",
            "search": actor_query,
            "language": "en",
            "uselang": "en",
            "type": "item",
            "limit": 10,
            "format": "json",
        }
    )
    request = Request(
        f"{WIKIDATA_API_URL}?{params}",
        headers={"User-Agent": "MovieRecommenderDemo/1.0 (Streamlit app)"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            results = json.load(response).get("search", [])
    except HTTPError as error:
        raise RuntimeError(
            f"Wikidata trả về lỗi HTTP {error.code}. Vui lòng thử lại sau."
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Không thể kết nối Wikidata. Hãy kiểm tra kết nối mạng: {error.reason}"
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Wikidata trả về dữ liệu không hợp lệ.") from error

    return [
        {
            "id": person["id"],
            "name": person.get("label", person["id"]),
            "description": person.get("description", ""),
        }
        for person in results
    ]


def get_wikidata_movie_credits(person_id):
    if not person_id.startswith("Q") or not person_id[1:].isdecimal():
        raise RuntimeError("Mã diễn viên từ Wikidata không hợp lệ.")
    query = f"""
    SELECT DISTINCT ?movieLabel ?releaseDate ?roleLabel WHERE {{
      VALUES ?person {{ wd:{person_id} }}
      ?movie p:P161 ?castStatement.
      ?castStatement ps:P161 ?person.
      OPTIONAL {{ ?movie wdt:P577 ?releaseDate. }}
      OPTIONAL {{ ?castStatement pq:P453 ?role. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,vi". }}
    }}
    ORDER BY DESC(?releaseDate)
    LIMIT 100
    """
    bindings = wikidata_query(query).get("results", {}).get("bindings", [])
    return [
        {
            "Phim": movie["movieLabel"]["value"],
            "Ngày phát hành": movie.get("releaseDate", {}).get("value", "")[:10],
            "Vai diễn": movie.get("roleLabel", {}).get("value", ""),
        }
        for movie in bindings
    ]


algo, names, movie_genres = load_all()
ts = algo.trainset


def recommend(raw_uid, n=10):
    uid = ts.to_inner_uid(str(raw_uid))
    seen = {i for i, _ in ts.ur[uid]}
    out = []
    for iid in ts.all_items():
        if iid not in seen:
            raw = ts.to_raw_iid(iid)
            out.append((names[int(raw)], round(algo.predict(str(raw_uid), raw).est, 2)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:n]


def liked(raw_uid, n=5):
    uid = ts.to_inner_uid(str(raw_uid))
    top = sorted(ts.ur[uid], key=lambda x: x[1], reverse=True)[:n]
    return [(names[int(ts.to_raw_iid(i))], r) for i, r in top]


st.title("🎬 Gợi ý phim - Nhóm 5 Super")
st.caption("Mô hình SVD (Surprise) huấn luyện trên MovieLens 100K")

tab_recommend, tab_movies, tab_actors = st.tabs(
    ["Gợi ý cho bạn", "Tìm phim", "Tìm diễn viên"]
)

with tab_recommend:
    uid = st.number_input(
        "Mã người dùng (1-943)", min_value=1, max_value=943, value=42, step=1
    )
    if st.button("Gợi ý"):
        try:
            st.subheader("Phim người này đã chấm cao nhất")
            st.table(pd.DataFrame(liked(uid), columns=["Phim", "Điểm đã chấm"]))
            st.subheader("Top-10 phim gợi ý")
            st.table(pd.DataFrame(recommend(uid), columns=["Phim", "Điểm dự đoán"]))
        except ValueError:
            st.error("Người dùng này không có trong tập huấn luyện.")

with tab_movies:
    with st.form("movie_search_form"):
        selected_genres = st.multiselect(
            "Chọn thể loại (có thể chọn nhiều)",
            options=[genre for genre in MOVIE_GENRES if genre != "unknown"],
            format_func=lambda genre: f"{GENRE_LABELS[genre]} ({genre})",
        )
        movie_query = st.text_input(
            "Hoặc nhập tên phim / thể loại",
            placeholder="Ví dụ: Toy Story, phim hành động, hài",
        ).strip()
        search_movies = st.form_submit_button("Tìm phim")

    if search_movies:
        normalized_query = normalize_text(movie_query)
        for prefix in ("phim the loai ", "the loai ", "phim "):
            if normalized_query.startswith(prefix):
                normalized_query = normalized_query[len(prefix):]
                break

        typed_genre = GENRE_ALIASES.get(normalized_query)
        required_genres = set(selected_genres)
        if typed_genre:
            required_genres.add(typed_genre)

        title_query = "" if typed_genre else normalized_query
        if not required_genres and not title_query:
            st.info("Hãy chọn ít nhất một thể loại hoặc nhập tên phim cần tìm.")
        else:
            matches = [
                (title, movie_genres[movie_id])
                for movie_id, title in names.items()
                if required_genres.issubset(set(movie_genres[movie_id]))
                and (
                    not title_query
                    or title_query in normalize_text(title)
                )
            ]
            if required_genres:
                genre_labels = ", ".join(sorted(required_genres))
                st.write(f"Phim thuộc tất cả thể loại: {genre_labels}")

            if matches:
                st.write(f"Tìm thấy {len(matches)} phim:")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Tên phim": title,
                                "Thể loại": ", ".join(genres) or "Chưa phân loại",
                            }
                            for title, genres in matches
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info(
                    "Không tìm thấy phim phù hợp trong MovieLens 100K. "
                    "Nếu chọn nhiều thể loại, phim cần thuộc tất cả thể loại đó."
                )

with tab_actors:
    st.caption(
        "Tra cứu miễn phí qua Wikidata, không cần API key. "
        "Cần kết nối internet; dữ liệu phim có thể không đầy đủ với một số diễn viên."
    )

    with st.form("actor_search_form"):
        actor_query = st.text_input(
            "Nhập tên diễn viên",
            placeholder="Ví dụ: Tom Hanks",
        ).strip()
        search_actors = st.form_submit_button("Tìm diễn viên")

    if search_actors:
        if not actor_query:
            st.session_state.pop("actor_search_query", None)
            st.session_state.pop("selected_actor_id", None)
            st.info("Hãy nhập tên diễn viên cần tìm.")
        else:
            st.session_state["actor_search_query"] = actor_query
            st.session_state.pop("selected_actor_id", None)

    saved_actor_query = st.session_state.get("actor_search_query", "")
    if saved_actor_query:
        try:
            people = search_wikidata_people(saved_actor_query)
            if not people:
                st.info("Không tìm thấy người phù hợp trên Wikidata.")
            else:
                people_by_id = {person["id"]: person for person in people}
                selected_person_id = st.selectbox(
                    "Chọn diễn viên",
                    options=list(people_by_id),
                    format_func=lambda person_id: (
                        f"{people_by_id[person_id]['name']} — "
                        f"{people_by_id[person_id]['description']}"
                        if people_by_id[person_id]["description"]
                        else people_by_id[person_id]["name"]
                    ),
                    key="selected_actor_id",
                )
                movies = get_wikidata_movie_credits(selected_person_id)
                if movies:
                    st.write("Các phim được Wikidata ghi nhận:")
                    st.dataframe(
                        pd.DataFrame(movies),
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    st.info(
                        "Wikidata chưa ghi nhận phim nào cho người này. "
                        "Dữ liệu có thể chưa đầy đủ."
                    )
        except (RuntimeError, KeyError, TypeError) as error:
            st.error(str(error))
