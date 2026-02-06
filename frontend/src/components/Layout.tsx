import { NavLink, Outlet } from 'react-router-dom'

export default function Layout() {
  return (
    <div className="layout">
      <nav className="navbar">
        <span className="brand">MDIMG</span>
        <NavLink to="/" className={({ isActive }) => isActive ? 'active' : ''}>
          Upload
        </NavLink>
        <NavLink to="/runs" className={({ isActive }) => isActive ? 'active' : ''}>
          Runs
        </NavLink>
      </nav>
      <main className="main-content">
        <Outlet />
      </main>
      <footer className="footer">
        MDIMG — Medical Image Quality Assurance &middot; Research use only
      </footer>
    </div>
  )
}
